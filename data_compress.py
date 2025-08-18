#!/usr/bin/env python3

import os
import shutil
import subprocess
import logging
import time
import concurrent.futures
from datetime import datetime

# ROOT can be /databf/nenufar-nri/LT02/2023 or /2023/01, /2024/03, etc.
ROOT = "/databf/nenufar-nri/LT02/2023/01"
PIPE_DIR = "/home/xzhang/software/exo_img_pipe/"
DP3_PARSET = os.path.join(PIPE_DIR, "templates/DPPP-average.parset")
CUTOFF = datetime.strptime("20231209", "%Y%m%d")

# Logging
log_filename = "compress_l1_parallel.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])

def get_obs_dirs(root):
    obs_dirs = []
    for dirpath, dirnames, _ in os.walk(root):
        if os.path.basename(dirpath) == "L1":
            obs_dir = os.path.dirname(dirpath)
            obs_name = os.path.basename(obs_dir)
            if len(obs_name) >= 8 and obs_name[:8].isdigit():
                obs_dirs.append((obs_name, obs_dir))
    return sorted(obs_dirs)

def compress_observation(obs_tuple):
    obs_name, obs_dir = obs_tuple
    try:
        obs_date_str = obs_name[:8]
        obs_date = datetime.strptime(obs_date_str, "%Y%m%d")
    except ValueError:
        logging.warning(f"Skipping {obs_name}: invalid date format")
        return

    l1_dir = os.path.join(obs_dir, "L1")
    if not os.path.isdir(l1_dir):
        logging.info(f"No L1 folder in {obs_name}, skipping.")
        return

    out_dir = os.path.join(obs_dir, "L1_compressed")
    os.makedirs(out_dir, exist_ok=True)

    # Copy diagnostics and logs
    for subdir in ["diagnostics", "logs"]:
        src = os.path.join(l1_dir, subdir)
        dst = os.path.join(out_dir, subdir)
        if os.path.isdir(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)

    # Process each MS file
    success = True
    for fname in os.listdir(l1_dir):
        if fname.startswith("SB") and fname.endswith(".MS"):
            ms_path = os.path.join(l1_dir, fname)
            ms_out = os.path.join(out_dir, f"{fname[:-3]}_com.MS")

            if os.path.exists(ms_out):
                logging.info(f"{ms_out} already exists, skipping.")
                continue

            if obs_date < CUTOFF:
                cmd = [
                    "DP3", DP3_PARSET,
                    f"msin={ms_path}",
                    f"msout={ms_out}",
                    "msout.storagemanager=dysco",
                    "avg.freqstep=12"
                ]
            else:
                cmd = [
                    "DP3", DP3_PARSET,
                    f"msin={ms_path}",
                    f"msout={ms_out}",
                    "msout.storagemanager=dysco",
                    "avg.freqstep=2",
                    "avg.timestep=2"
                ]

            try:
                logging.info(f"Running: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                logging.info(f"Compressed {fname} -> {ms_out}")
            except subprocess.CalledProcessError as e:
                logging.error(f"DP3 failed on {fname}: {e}")
                success = False

    # If all MS files processed, delete original L1 folder
    if success:
        try:
            shutil.rmtree(l1_dir)
            logging.info(f"Removed original L1 folder for {obs_name}")
        except Exception as e:
            logging.error(f"Failed to delete L1 folder: {e}")

if __name__ == "__main__":
    obs_list = get_obs_dirs(ROOT)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        executor.map(compress_observation, obs_list)
