#!/usr/bin/env python3

import os
import shutil
import subprocess
import logging
import time
import concurrent.futures
from datetime import datetime

# ROOT can be /databf/nenufar-nri/LT02/2023 or /2023/01, /2024/03, etc.
ROOT = "/databf/nenufar-nri/LT02/2023/08"
PIPE_DIR = "/home/xzhang/software/exo_img_pipe/"
DP3_PARSET = os.path.join(PIPE_DIR, "templates/DPPP-average.parset")
CUTOFF = datetime.strptime("20231209", "%Y%m%d")

# Logging
log_filename = "compress_l1_parallel.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])

def get_obs_dirs(root):
    obs_dirs = []
    root_basename = os.path.basename(os.path.normpath(root))

    if root_basename.isdigit():
        if len(root_basename) == 4:
            # ROOT is a year folder like .../2023
            for month in sorted(os.listdir(root)):
                month_path = os.path.join(root, month)
                if not os.path.isdir(month_path):
                    continue

                for obs in sorted(os.listdir(month_path)):
                    obs_path = os.path.join(month_path, obs)
                    l1_path = os.path.join(obs_path, "L1")
                    if os.path.isdir(l1_path):
                        obs_dirs.append((obs, obs_path))

        elif len(root_basename) == 2:
            # ROOT is a month folder like .../2023/01
            for obs in sorted(os.listdir(root)):
                obs_path = os.path.join(root, obs)
                l1_path = os.path.join(obs_path, "L1")
                if os.path.isdir(l1_path):
                    obs_dirs.append((obs, obs_path))

    else:
        logging.warning(f"Could not determine folder level from ROOT name: {root}")

    return obs_dirs

def can_delete_folder(folder_path):
    try:
        test_path = os.path.join(folder_path, ".tmp_permission_check")
        with open(test_path, "w") as f:
            f.write("test")
        os.remove(test_path)
        return True
    except Exception as e:
        logging.warning(f"No permission to modify {folder_path}: {e}")
        return False

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
    
    if not can_delete_folder(l1_dir):
        logging.info(f"Skipping {obs_name}: no write/delete permission in L1 folder.")
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        executor.map(compress_observation, obs_list)
