#!/usr/bin/env python3

from prefect import flow, task
import subprocess
import os, argparse, glob
from templates.Find_Bad_MAs_template import find_bad_MAs
from datetime import datetime

# Set file locations
watch_dir = "/databf/nenufar-nri/LT02/2023/03/"

preprocess_dir = "/databf/nenufar-nri/LT02/"
postprocess_dir = "/data/xzhang/exo_img/"
pipe_dir = "/home/xzhang/software/exo_img_pipe/"
lockfile = "/home/xzhang/software/exo_img_pipe/lock.file"

# Calibrators
CALIBRATORS = ['CYG_A', 'CAS_A', 'TAU_A', 'VIR_A']

# How many SB per processing chunk
chunk_num = 12

# How many channels per SB
chan_per_SB = 12

# Task 0. Find un-processed data directories

@task
def check_new_data(watch_dir: str, postprocess_dir: str) -> list:
    avai_dir = glob.glob(watch_dir + "*")
    avai_data = [f.split('/')[-1] for f in avai_dir]
    
    processed_dir = glob.glob(postprocess_dir + "*")
    processed_data = [f.split('/')[-1] for f in processed_dir]

    # Filtering out the data that's already processed and not in CALIBRATORS
    unprocessed_data = [data for data in avai_data if data not in processed_data and not any(cal in data for cal in CALIBRATORS)]

    return unprocessed_data

# Task 1. Moving target and calibrator data from /databf to /data where we do the processing

@task(log_prints=True)
def copy_astronomical_data(exo_dir: str):
    # Parse year and month from exo_dir
    parts = exo_dir.split("_")
    start_time = parts[0] + "_" + parts[1]
    end_time = parts[2] + "_" + parts[3]

    year = parts[0][:4]
    month = parts[0][4:6]
    day = parts[0][6:8]

    data_date = datetime(int(year), int(month), int(day))

    # Construct source and destination directory paths
    pre_target_dir = os.path.join(preprocess_dir, year, month, exo_dir)
    post_target_dir = os.path.join(postprocess_dir, exo_dir)
    
    # Search for calibrator directory
    base_cal_dir = os.path.join(preprocess_dir, year, month)
    potential_dirs = [d for d in os.listdir(base_cal_dir) if any(cal in d for cal in CALIBRATORS)]
    
    valid_cal_dirs = []
    for dir in potential_dirs:
        parts = dir.split("_")
        dir_start_time = parts[0] + "_" + parts[1]
        dir_end_time = parts[2] + "_" + parts[3]
        if dir_start_time == end_time or dir_end_time == start_time:
            exo_files_count = len(os.listdir(os.path.join(pre_target_dir, "L1")))
            cal_files_count = len(os.listdir(os.path.join(base_cal_dir, dir, "L1")))
            if exo_files_count == cal_files_count:
                valid_cal_dirs.append(dir)
    
    if len(valid_cal_dirs) < 1:
        cmd = f"mkdir {post_target_dir}"
        subprocess.run(cmd, shell=True, check=True)
        raise FileNotFoundError(f"Calibrator data not found.")

    cal_dir = valid_cal_dirs[0]

    # data after a specific date would have scp completion marker.
    comparison_date = datetime(2023, 10, 1)
    scp_marker_target = pre_target_dir + '/L1/' + 'copper_copy_done'
    scp_marker_cali = base_cal_dir + '/' + cal_dir + '/L1/' + 'copper_copy_done'

    if data_date > comparison_date:
        if not os.path.exists(scp_marker_cali) or not os.path.exists(scp_marker_target):
            raise FileNotFoundError(f"SCP marker not found. COPPER scp ongoing. Please wait.")

    if os.path.exists(f"{postprocess_dir}/{cal_dir}"):
        print("Calibrator data already processed. "+ postprocess_dir + '/' + cal_dir)
        cali_check = True
    else:
        cmd = f"rsync -av --progress {base_cal_dir}/{cal_dir}/L1/ {postprocess_dir}/{cal_dir}"
        subprocess.run(cmd, shell=True, check=True)
        cali_check = False

    # Use rsync to copy data
    cmd = f"rsync -av --progress {pre_target_dir}/L1/ {post_target_dir}"
    subprocess.run(cmd, shell=True, check=True)

    for cal in CALIBRATORS:
        if cal in cal_dir:
            return cal, cal_dir, cali_check
    raise ValueError("Calibrator not found in the valid cal_dir.")

# Task 2. Do a testing round of calibration to find the bad Mini Arrays

@task
def identify_bad_mini_arrays(cal: str, cal_dir: str) -> str:
    # Step 1: Set the environment
    cmd = "use DP3"
    subprocess.run(cmd, shell=True, check=True)

    # Step 2: Run DP3 DPPP-aoflagger.parset command
    cali_SB = glob.glob(postprocess_dir + cal_dir + '/SB*.MS')
    cali_SB.sort()

    # Determine the number of full chunks of chunk_num we can form
    num_chunks = len(cali_SB) // chunk_num

    for i in range(num_chunks):
        # Extract the ith chunk of chunk_num file names
        chunk = cali_SB[i * chunk_num: (i + 1) * chunk_num]

        # Create the msin string by joining the chunk with commas
        SB_str = ", ".join(chunk)

        # Construct the output file name using the loop index (i+1)
        MSB_filename = f"{postprocess_dir}/{cal_dir}/MSB{str(i).zfill(2)}.MS"

        # Construct the command string with the msin argument and the msout argument
        cmd_flagchan = f"DP3 {pipe_dir}/templates/DPPP-flagchan.parset msin=[{SB_str}] msout={MSB_filename}"
        
        # Run the command using subprocess
        subprocess.run(cmd_flagchan, shell=True, check=True)

        # Construct the command string with the msin argument and the msout argument
        cmd_aoflagger = f"DP3 {pipe_dir}/templates/DPPP-aoflagger.parset msin={MSB_filename}"

        # Run the command using subprocess
        subprocess.run(cmd_aoflagger, shell=True, check=True)
        
        # Read the template file
        with open(f'{pipe_dir}/templates/calibrator.toml', 'r') as template_file:
            template_content = template_file.read()

        # Perform the replacements
        cali_model = f'{pipe_dir}/cal_models/{cal}_lcs.skymodel'

        # Replace placeholders in the template content
        modified_content = template_content.replace('CALI_MODEL', cali_model)
        modified_content = modified_content.replace('CHAN_PER_SB', str(chan_per_SB))

        # Write the modified content to a new file
        with open(f'{postprocess_dir}/{cal_dir}/cali.toml', 'w') as cali_file:
            cali_file.write(modified_content)

        cmd_cali = f"calpipe {postprocess_dir}/{cal_dir}/cali.toml {MSB_filename}"
        subprocess.run(cmd_cali, shell=True, check=True)

    # Step 3: Call the imported function directly
    bad_MAs = find_bad_MAs(f"{postprocess_dir}/{cal_dir}/")

    return bad_MAs


@flow(name="EXO_IMG PIPELINE", log_prints=True)
def exo_pipe(exo_dir):
    with open(lockfile, "w") as f:
        f.write("Processing ongoing")

    cal, cal_dir, cali_check = copy_astronomical_data(exo_dir)

    # Has calibrator been processed already?
    if cali_check == False:
        bad_MAs = identify_bad_mini_arrays(cal, cal_dir)

    os.remove(lockfile)

@flow(name='Check Flow', log_prints=True)
def check_flow(dir_to_check):
    if os.path.exists(lockfile):
        exit()

    new_data = check_new_data(watch_dir, postprocess_dir)

    if len(new_data) > 0:
        # Trigger the main flow
        for unprocessed_data in new_data:
            exo_pipe(unprocessed_data)

if __name__ == "__main__":
    check_flow.serve(name="check-flow", interval=10800)