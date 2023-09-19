#!/usr/bin/env python3

from prefect import Flow, task
import subprocess
import os

CALIBRATORS = ['CYG_A', 'CAS_A', 'TAU_A', 'VIR_A']

@task
def copy_astronomical_data(exo_dir: str):
    # Parse year and month from exo_dir
    parts = exo_dir.split("_")
    start_time = parts[0] + "_" + parts[1]
    end_time = parts[2] + "_" + parts[3]

    year = parts[0][:4]
    month = parts[0][4:6]
    
    # Construct source and destination directory paths
    source_dir = os.path.join("/databf/nenufar-nri/LT02/", year, month, exo_dir)
    destination_dir = os.path.join("/data/xzhang/exo_img/", exo_dir)

    # Ensure destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Use rsync to copy data
    cmd = f"rsync -av --progress {source_dir}/ {destination_dir}/"
    subprocess.run(cmd, shell=True, check=True)

    # Search for calibrator directory
    base_cal_dir = os.path.join("/databf/nenufar-nri/LT02/", year, month)
    potential_dirs = [d for d in os.listdir(base_cal_dir) if any(cal in d for cal in CALIBRATORS)]
    
    valid_cal_dirs = []
    for dir in potential_dirs:
        parts = dir.split("_")
        dir_start_time = parts[0] + "_" + parts[1]
        dir_end_time = parts[2] + "_" + parts[3]
        if dir_start_time == end_time or dir_end_time == start_time:
            exo_files_count = len(os.listdir(os.path.join(source_dir, "L1")))
            cal_files_count = len(os.listdir(os.path.join(base_cal_dir, dir, "L1")))
            if exo_files_count == cal_files_count:
                valid_cal_dirs.append(dir)
    
    if len(valid_cal_dirs) != 1:
        raise ValueError(f"Found {len(valid_cal_dirs)} valid calibrator directories, expected exactly 1.")

    cal_dir = valid_cal_dirs[0]
    cmd = f"rsync -av --progress {base_cal_dir}/{cal_dir}/ {destination_dir}/{cal_dir}/"
    subprocess.run(cmd, shell=True, check=True)

    for cal in CALIBRATORS:
        if cal in cal_dir:
            return cal
    raise ValueError("Calibrator not found in the valid cal_dir.")


with Flow("Astronomical Data Pipeline") as flow:
    exo_dir = "20230918_193000_20230918_234900_HD_189733_TRACKING"  # For testing
    copy_astronomical_data(exo_dir)

# Run the flow
flow.run()
