#!/usr/bin/env python3

from prefect import flow, task
from prefect.states import Completed
import subprocess
import os, glob
from astropy.io import fits
import numpy as np
from dask import delayed, compute
from casatools import table
from templates.Find_Bad_MAs_template import find_bad_MAs
from templates.Make_Target_List_template import make_target_list
from templates.Plot_target_distri_template import plot_target_distribution
from templates.Noise_esti_template import generate_noise_map_v, calculate_noise_for_window, apply_gaussian_filter
from templates.Noise_esti_template import generate_and_save_weight_map_v, source_detection, generate_noise_map_i, generate_and_save_weight_map_i
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

###### Initial settings ######

# Set file locations
watch_dir = "/databf/nenufar-nri/LT02/202?/??/*HD_189733*"

preprocess_dir = "/databf/nenufar-nri/LT02/"
postprocess_dir = "/data/xzhang/exo_img/"
pipe_dir = "/home/xzhang/software/exo_img_pipe/"
lockfile = "/home/xzhang/software/exo_img_pipe/lock.file"
singularity_file = "/home/cyril.tasse/DDFSingularity/ddf_dev2.sif"
skip_file = "/home/xzhang/software/exo_img_pipe/templates/skip.txt"

# Calibrators
CALIBRATORS = ['CYG_A', 'CAS_A', 'TAU_A', 'VIR_A']

# How many SB per processing chunk
# chunk_num = 12

# How many channels per SB
chan_per_SB_origin = 12
ave_chan = 4
chan_per_SB = int(chan_per_SB_origin/ave_chan)

# chan_per_SB = 12

# Avoid bad channel making KMS hang
# bin_per_MSB = chunk_num // 3

# the lowest SB we use
SB_min = 106 # 92
SB_ave_kms = 2

# The region file we use for A-team removal
region_file = "/home/xzhang/software/exo_img_pipe/regions/CygA.reg"

# Window and SNR threshold for matched filtering
direction_threshold = 6
direction_threshold_target = 5
dynamic_threshold = 6
dynamic_threshold_target = 5
# snr_threshold = 7
# snr_threshold_target = 6
time_windows = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
freq_windows = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
# 30 sec (5 min) 
# 600 kHz (6 MHz)

###### Lock the flow runs when data processing is ongoing ######

# class LockExitException(Exception):
#     pass

###### Here are the tasks (aka functions doing the job) ######

# Task 0. Find un-processed data directories

@task(log_prints=True)
def check_new_data(watch_dir: str, postprocess_dir: str) -> list:
    avai_dir = glob.glob(watch_dir + "*/L1")
    avai_data = [f.split('/')[-2] for f in avai_dir]
    
    processed_dir = glob.glob(postprocess_dir + "*")
    processed_data = [f.split('/')[-1] for f in processed_dir]
    skip_data = np.genfromtxt(skip_file, dtype='str')

    # Filtering out the data that's already processed and not in CALIBRATORS
    unprocessed_data = [data for data in avai_data if data not in processed_data and not any(cal in data for cal in CALIBRATORS) and data not in skip_data]

    return unprocessed_data

# Task 1. Moving target and calibrator data from /databf to /data where we do the processing

@task(log_prints=True)
def copy_calibrator_data(exo_dir: str):
    # Parse year and month from exo_dir
    parts = exo_dir.split("_")
    start_time = parts[0] + "_" + parts[1]
    end_time = parts[2] + "_" + parts[3]

    year = parts[0][:4]
    month = parts[0][4:6]
    # day = parts[0][6:8]

    start_date = parts[0]
    end_date = parts[2]
    

    # data_date = datetime(int(year), int(month), int(day))

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
    
    # If no valid calibrator directory is found, get the calibrator directory with the same date
    if len(valid_cal_dirs) < 1: 
        for dir in potential_dirs:
            parts = dir.split("_")
            dir_start_date = parts[0] 
            dir_end_date = parts[2]
            if dir_start_date == start_date or dir_end_date == end_date:
                exo_files_count = len(os.listdir(os.path.join(pre_target_dir, "L1")))
                cal_files_count = len(os.listdir(os.path.join(base_cal_dir, dir, "L1")))
                if exo_files_count == cal_files_count:
                    valid_cal_dirs.append(dir)

    # No calibrator data found for that day, raise error
    if len(valid_cal_dirs) < 1:
        cmd = f"mkdir {post_target_dir}"
        subprocess.run(cmd, shell=True, check=True)
        raise FileNotFoundError(f"Calibrator data not found.")

    cal_dir = valid_cal_dirs[0]

    # data after a specific date would have scp completion marker.
    # 2023-10-04: Cedric modified preprocessing code so this part is no longer necessary.
    # comparison_date = datetime(2023, 10, 1)
    # scp_marker_target = pre_target_dir + '/L1/' + 'copper_copy_done'
    # scp_marker_cali = base_cal_dir + '/' + cal_dir + '/L1/' + 'copper_copy_done'

    # if data_date > comparison_date:
    #     if not os.path.exists(scp_marker_cali) or not os.path.exists(scp_marker_target):
    #         raise FileNotFoundError(f"SCP marker not found. COPPER scp ongoing. Please wait.")

    if os.path.exists(f"{postprocess_dir}/{cal_dir}"):
        print("Calibrator data already processed. "+ postprocess_dir + '/' + cal_dir)
        cali_check = True
    else:
        cmd = f"rsync -av --progress {base_cal_dir}/{cal_dir}/L1/ {postprocess_dir}/{cal_dir}"
        subprocess.run(cmd, shell=True, check=True)
        cali_check = False

    for cal in CALIBRATORS:
        if cal in cal_dir:
            return cal, cal_dir, cali_check
    raise ValueError("Calibrator not found in the valid cal_dir.")

# Task 1.5. Copy target data
@task(log_prints=True)
def copy_target_data(exo_dir: str):
    # Parse year and month from exo_dir
    parts = exo_dir.split("_")

    year = parts[0][:4]
    month = parts[0][4:6]

    # Construct source and destination directory paths
    pre_target_dir = os.path.join(preprocess_dir, year, month, exo_dir)
    post_target_dir = os.path.join(postprocess_dir, exo_dir)
    # Use rsync to copy data
    cmd = f"rsync -av --progress {pre_target_dir}/L1/ {post_target_dir}"
    subprocess.run(cmd, shell=True, check=True)


# Task 2. Do a testing round of calibration to find the bad Mini Arrays

@task(log_prints=True)
def identify_bad_mini_arrays(cal: str, cal_dir: str) -> str:
    # Step 1: Set the environment
    # cmd = "use DP3"
    # subprocess.run(cmd, shell=True, check=True)

    # Step 2: Run DP3 DPPP-aoflagger.parset command
    cali_SB_0 = glob.glob(postprocess_dir + cal_dir + '/SB*.MS')
    cali_SB = [f for f in cali_SB_0 if int(f.split('/SB')[1].split('.MS')[0]) > SB_min]
    cali_SB.sort()

    # Now we need to make sure that the number of SB is a multiple of SB_ave_kms. We can remove the first few SBs if necessary
    num_SB = len(cali_SB)
    num_remove = num_SB % SB_ave_kms
    cali_SB = cali_SB[num_remove:]

    # Read the template file
    with open(f'{pipe_dir}/templates/bad_MA.toml', 'r') as template_file:
        template_content = template_file.read()

    # Perform the replacements
    cali_model = f'{pipe_dir}/cal_models/{cal}_lcs.skymodel'

    # Replace placeholders in the template content
    modified_content = template_content.replace('CALI_MODEL', cali_model)
    modified_content = modified_content.replace('CHAN_PER_SB', str(chan_per_SB))

    # Write the modified content to a new file
    with open(f'{postprocess_dir}/{cal_dir}/cali.toml', 'w') as cali_file:
        cali_file.write(modified_content)

    # Determine the number of full chunks of chunk_num we can form
    # num_chunks = len(cali_SB) // chunk_num

    # cmd_sky_dir = f"mkdir {postprocess_dir}/{cal_dir}/sky_models/"
    # subprocess.run(cmd_sky_dir, shell=True, check=True)

    # Group files by their tens place
    groups = {}
    for f in cali_SB:
        sb_number = int(f.split('/SB')[1].split('.MS')[0])
        tens_place = sb_number // 10
        if tens_place not in groups:
            groups[tens_place] = []
        groups[tens_place].append(f)

    for tens_place, chunk in groups.items():
        # # Extract the ith chunk of chunk_num file names
        # chunk = cali_SB[i * chunk_num: (i + 1) * chunk_num]

        # Create the msin string by joining the chunk with commas
        SB_str = ",".join(chunk)

        # Construct the output file name using the loop index (i+1)
        MSB_filename = f"{postprocess_dir}/{cal_dir}/MSB{str(tens_place).zfill(2)}.MS"

        # Construct the command string with the msin argument and the msout argument
        cmd_flagchan = f"DP3 {pipe_dir}/templates/DPPP-flagchan.parset msin=[{SB_str}] msout={MSB_filename} avg.freqstep={ave_chan}"
        subprocess.run(cmd_flagchan, shell=True, check=True)

    # Stack the GSB.MS
    msb_files = glob.glob(f"{postprocess_dir}/{cal_dir}/MSB*.MS")
    msb_files.sort()
    msb_files_str = ",".join(msb_files)
    cmd_stack = f"DP3 {pipe_dir}/templates/DPPP-stack.parset msin=[{msb_files_str}] msout={postprocess_dir}/{cal_dir}/GSB.MS"
    subprocess.run(cmd_stack, shell=True, check=True)

    # Construct the command string with the msin argument and the msout argument
    cmd_aoflagger = f"DP3 {pipe_dir}/templates/DPPP-aoflagger.parset msin={postprocess_dir}/{cal_dir}/GSB.MS flag.strategy={pipe_dir}/templates/Nenufar64C1S.lua"
    subprocess.run(cmd_aoflagger, shell=True, check=True)

    cmd_cali = f"calpipe {postprocess_dir}/{cal_dir}/cali.toml {postprocess_dir}/{cal_dir}/GSB.MS"
    subprocess.run(cmd_cali, shell=True, check=True)

    # cmd_makesky = f"mkdir {postprocess_dir}/{cal_dir}/sky_models/MSB{str(i).zfill(2)}/"
    # subprocess.run(cmd_makesky, shell=True, check=True)

    cmd_movesky = f"mv {postprocess_dir}/{cal_dir}/GSB.MS/sky_model {postprocess_dir}/{cal_dir}/"
    subprocess.run(cmd_movesky, shell=True, check=True)

    # Step 3: Call the imported function directly
    bad_MAs = find_bad_MAs(f"{postprocess_dir}/{cal_dir}/")

    # Step 4: Remove the testing MSB
    cmd_remo_GSB = f"rm -rf {postprocess_dir}/{cal_dir}/GSB.MS"
    subprocess.run(cmd_remo_GSB, shell=True, check=True)

    return bad_MAs

# Task 3. Calibration with A team

@task(log_prints=True)
def calibration_Ateam(cal: str, cal_dir: str, bad_MAs: str):
    # Step 1: Set the environment
    # cmd = "use DP3"
    # subprocess.run(cmd, shell=True, check=True)

    # Step 2: Run DP3 DPPP-aoflagger.parset command
    cali_SB_0 = glob.glob(postprocess_dir + cal_dir + '/SB*.MS')
    cali_SB = [f for f in cali_SB_0 if int(f.split('/SB')[1].split('.MS')[0]) > SB_min]
    cali_SB.sort()

    # Now we need to make sure that the number of SB is a multiple of SB_ave_kms. We can remove the first few SBs if necessary
    num_SB = len(cali_SB)
    num_remove = num_SB % SB_ave_kms
    cali_SB = cali_SB[num_remove:]

    # Use casatools to get the list of antennas observed
    tb = table()
    tb.open(cali_SB[0] + '/ANTENNA')
    antennas = tb.getcol('NAME')
    tb.close()

    # save the list of antennas to a file, seperated by commas
    with open(f'{postprocess_dir}/{cal_dir}/All_MAs.txt', 'w') as f:
        f.write(','.join(antennas))

    # # Flag the bad MAs
    # with open(f'{pipe_dir}/templates/DPPP-flagant.parset', 'r') as template_flag:
    #     flag_content = template_flag.read()

    # modified_flag_content = flag_content.replace('MA_TO_FLAG', bad_MAs)

    # # Write the modified content to a new file
    # with open(f'{postprocess_dir}/{cal_dir}/DPPP-flagant.parset', 'w') as flag_file:
    #     flag_file.write(modified_flag_content)

    # remove the bad MAs rather than flagging them
    with open(f'{pipe_dir}/templates/DPPP-removeant.parset', 'r') as template_remove:
        remove_content = template_remove.read()

    modified_remove_content = remove_content.replace('MA_TO_REMOVE', bad_MAs)

    # Write the modified content to a new file
    with open(f'{postprocess_dir}/{cal_dir}/DPPP-removeant.parset', 'w') as remove_file:
        remove_file.write(modified_remove_content)

    # Stack the GSB.MS
    msb_files = glob.glob(f"{postprocess_dir}/{cal_dir}/MSB*.MS")
    msb_files.sort()
    msb_files_str = ",".join(msb_files)
    cmd_stack = f"DP3 {pipe_dir}/templates/DPPP-stack.parset msin=[{msb_files_str}] msout={postprocess_dir}/{cal_dir}/GSB.MS"
    subprocess.run(cmd_stack, shell=True, check=True)

    # cmd_flagMA = f"DP3 {postprocess_dir}/{cal_dir}/DPPP-flagant.parset msin={postprocess_dir}/{cal_dir}/GSB.MS"
    # subprocess.run(cmd_flagMA, shell=True, check=True)

    # remove the bad MAs rather than flagging them, notice that we need to generate a new GSB.MSB file
    cmd_removeMA = f"DP3 {postprocess_dir}/{cal_dir}/DPPP-removeant.parset msin={postprocess_dir}/{cal_dir}/GSB.MS msout={postprocess_dir}/{cal_dir}/GSB.MSB"
    subprocess.run(cmd_removeMA, shell=True, check=True)
    # remove original MSB
    cmd_remo_GSB = f"rm -rf {postprocess_dir}/{cal_dir}/GSB.MS"
    subprocess.run(cmd_remo_GSB, shell=True, check=True)
    # rename the new MSB
    cmd_rename_GSB = f"mv {postprocess_dir}/{cal_dir}/GSB.MSB {postprocess_dir}/{cal_dir}/GSB.MS"
    subprocess.run(cmd_rename_GSB, shell=True, check=True)

        # Construct the command string with the msin argument and the msout argument
    cmd_aoflagger = f"DP3 {pipe_dir}/templates/DPPP-aoflagger.parset msin={postprocess_dir}/{cal_dir}/GSB.MS flag.strategy={pipe_dir}/templates/Nenufar64C1S.lua"
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

    cmd_copysky = f"cp -rf {postprocess_dir}/{cal_dir}/sky_model {postprocess_dir}/{cal_dir}/GSB.MS/"
    subprocess.run(cmd_copysky, shell=True, check=True)

    cmd_cali = f"calpipe {postprocess_dir}/{cal_dir}/cali.toml {postprocess_dir}/{cal_dir}/GSB.MS"
    subprocess.run(cmd_cali, shell=True, check=True)

    # Remove the table files so they don't take up too much space!
    cmd_remo_table = f"rm -rf {postprocess_dir}/{cal_dir}/GSB.MS/table.* {postprocess_dir}/{cal_dir}/GSB.MS/pre_cal_flags.h5"
    subprocess.run(cmd_remo_table, shell=True, check=True)

    # remove the SB*.MS files
    cmd_remo_SB = f"rm -rf {postprocess_dir}/{cal_dir}/SB*.MS {postprocess_dir}/{cal_dir}/MSB*.MS"
    subprocess.run(cmd_remo_SB, shell=True, check=True)

# Task 4. Apply A-team calibration solution to target

@task(log_prints=True)
def apply_Ateam_solution(cal_dir: str, exo_dir: str, bad_MAs: str):
    # Step 1: Set the environment
    # cmd = "use DP3"
    # subprocess.run(cmd, shell=True, check=True)

    # Step 2: Run DP3 DPPP-aoflagger.parset command
    exo_SB_0 = glob.glob(postprocess_dir + exo_dir + '/SB*.MS')
    exo_SB = [f for f in exo_SB_0 if int(f.split('/SB')[1].split('.MS')[0]) > SB_min]
    exo_SB.sort()

    # Now we need to make sure that the number of SB is a multiple of SB_ave_kms. We can remove the first few SBs if necessary
    num_SB = len(exo_SB)
    num_remove = num_SB % SB_ave_kms
    exo_SB = exo_SB[num_remove:]

    # we need a list of antennas to be compared with the antennas in calibration
    # Use casatools to get the list of antennas observed
    tb = table()
    tb.open(exo_SB[0] + '/ANTENNA')
    exo_antennas = tb.getcol('NAME')
    tb.close()

    # read the list of antennas in calibration
    with open(f'{postprocess_dir}/{cal_dir}/All_MAs.txt', 'r') as f:
        cal_antennas = f.read().split(',')

    # find the antennas that are not in calibration
    remove_antennas = [ant for ant in exo_antennas if ant not in cal_antennas]

    # since we remove bad MAs rather than flag them, we need to combine the bad MAs and the antennas to be removed
    remove_antennas = list(set(remove_antennas + bad_MAs.split(',')))
    remove_antennas.sort()

    # only write the file when there are antennas to be removed
    if len(remove_antennas) > 0:
        remove_MA_names = ','.join(remove_antennas)
        # modify DPPP-removeant.parset in a similar way to DPPP-flagant.parset
        with open(f'{pipe_dir}/templates/DPPP-removeant.parset', 'r') as template_remove:
            remove_content = template_remove.read()

        modified_remove_content = remove_content.replace('MA_TO_REMOVE', remove_MA_names)

        # Write the modified content to a new file
        with open(f'{postprocess_dir}/{exo_dir}/DPPP-removeant.parset', 'w') as remove_file:
            remove_file.write(modified_remove_content)

    # # Flag the bad MAs
    # with open(f'{pipe_dir}/templates/DPPP-flagant.parset', 'r') as template_flag:
    #     flag_content = template_flag.read()

    # modified_flag_content = flag_content.replace('MA_TO_FLAG', bad_MAs)

    # # Write the modified content to a new file
    # with open(f'{postprocess_dir}/{exo_dir}/DPPP-flagant.parset', 'w') as flag_file:
    #     flag_file.write(modified_flag_content)

    # Determine the number of full chunks of chunk_num we can form
    # num_chunks = len(exo_SB) // chunk_num

    # Group files by their tens place
    groups = {}
    for f in exo_SB:
        sb_number = int(f.split('/SB')[1].split('.MS')[0])
        tens_place = sb_number // 10
        if tens_place not in groups:
            groups[tens_place] = []
        groups[tens_place].append(f)

    for tens_place, chunk in groups.items():
        # Extract the ith chunk of chunk_num file names
        # chunk = exo_SB[i * chunk_num: (i + 1) * chunk_num]

        # Create the msin string by joining the chunk with commas
        SB_str = ",".join(chunk)

        # Construct the output file name using the loop index (i+1)
        MSB_filename = f"{postprocess_dir}/{exo_dir}/MSB{str(tens_place).zfill(2)}.MS"

        # Construct the command string with the msin argument and the msout argument
        cmd_flagchan = f"DP3 {pipe_dir}/templates/DPPP-flagchan.parset msin=[{SB_str}] msout={MSB_filename} avg.freqstep={ave_chan}"
        subprocess.run(cmd_flagchan, shell=True, check=True)

    # Stack the GSB.MS
    msb_files = glob.glob(f"{postprocess_dir}/{exo_dir}/MSB*.MS")
    msb_files.sort()
    msb_files_str = ",".join(msb_files)
    cmd_stack = f"DP3 {pipe_dir}/templates/DPPP-stack.parset msin=[{msb_files_str}] msout={postprocess_dir}/{exo_dir}/GSB.MS"
    subprocess.run(cmd_stack, shell=True, check=True)

    # only run removeant when there are antennas to be removed
    if len(remove_antennas) > 0:
        cmd_removeMA = f"DP3 {postprocess_dir}/{exo_dir}/DPPP-removeant.parset msin={postprocess_dir}/{exo_dir}/GSB.MS msout={postprocess_dir}/{exo_dir}/GSB.MSB"
        subprocess.run(cmd_removeMA, shell=True, check=True)
        # remove original MSB
        cmd_remo_GSB = f"rm -rf {postprocess_dir}/{exo_dir}/GSB.MS"
        subprocess.run(cmd_remo_GSB, shell=True, check=True)
        # rename the new MSB
        cmd_rename_GSB = f"mv {postprocess_dir}/{exo_dir}/GSB.MSB {postprocess_dir}/{exo_dir}/GSB.MS"
        subprocess.run(cmd_rename_GSB, shell=True, check=True)

    # cmd_flagMA = f"DP3 {postprocess_dir}/{exo_dir}/DPPP-flagant.parset msin={postprocess_dir}/{exo_dir}/GSB.MS"
    # subprocess.run(cmd_flagMA, shell=True, check=True)

    # Construct the command string with the msin argument and the msout argument
    cmd_aoflagger = f"DP3 {pipe_dir}/templates/DPPP-aoflagger.parset msin={postprocess_dir}/{exo_dir}/GSB.MS flag.strategy={pipe_dir}/templates/Nenufar64C1S.lua"
    subprocess.run(cmd_aoflagger, shell=True, check=True)

    # Copy calibration solution
    cmd_copy_solution = f'cp {postprocess_dir}/{cal_dir}/GSB.MS/instrument_ddecal.h5 {postprocess_dir}/{exo_dir}/GSB.MS/instrument_dical.h5'
    subprocess.run(cmd_copy_solution, shell=True, check=True)

    # apply solution
    cmd_apply_solution = f'calpipe {pipe_dir}/templates/cali_tran.toml {postprocess_dir}/{exo_dir}/GSB.MS'
    subprocess.run(cmd_apply_solution, shell=True, check=True)

    # second round of aoflagger
    cmd_aoflagger = f"DP3 {pipe_dir}/templates/DPPP-aoflagger.parset msin={postprocess_dir}/{exo_dir}/GSB.MS msin.datacolumn=DI_DATA flag.strategy={pipe_dir}/templates/Nenufar64C1S.lua"
    subprocess.run(cmd_aoflagger, shell=True, check=True)

    # Now we average the GSB.MS into a smaller size
    cmd_avg = f"DP3 {pipe_dir}/templates/DPPP-average.parset msin={postprocess_dir}/{exo_dir}/GSB.MS msout={postprocess_dir}/{exo_dir}/GSB_avg.MS msin.datacolumn=DI_DATA avg.freqstep={chan_per_SB}"
    subprocess.run(cmd_avg, shell=True, check=True)

    # replace the GSB.MS with the averaged one
    cmd_repl = f"rm -rf {postprocess_dir}/{exo_dir}/GSB.MS"
    subprocess.run(cmd_repl, shell=True, check=True)

    cmd_repl = f"mv {postprocess_dir}/{exo_dir}/GSB_avg.MS {postprocess_dir}/{exo_dir}/GSB.MS"
    subprocess.run(cmd_repl, shell=True, check=True)

# Task 5. Subtract A-team from field

@task(log_prints=True)
def subtract_Ateam(exo_dir: str):
    # Step 1: Set the environment
    singularity_command = f"singularity exec -B/data/$USER {singularity_file}"

    # we need the number of exo_SB for the following steps
    exo_SB_0 = glob.glob(postprocess_dir + exo_dir + '/SB*.MS')
    exo_SB = [f for f in exo_SB_0 if int(f.split('/SB')[1].split('.MS')[0]) > SB_min]

    # Now we need to make sure that the number of SB is a multiple of SB_ave_kms. We can remove the first few SBs if necessary
    num_SB = len(exo_SB)
    num_remove = num_SB % SB_ave_kms
    exo_SB = exo_SB[num_remove:]

    # we need the number of beams for the following steps
    num_SB = len(exo_SB)
    num_beam = int(num_SB / SB_ave_kms)

    # modify the code to use GSB.MS, rather than multiple MSB???.MS files
    cmd_ddf = (
        f'DDF.py {pipe_dir}/templates/template_DI.parset --Data-MS {postprocess_dir}{exo_dir}/GSB.MS --Data-ColName DATA --Output-Name {postprocess_dir}{exo_dir}/Image_DI_Bis '
        f'--Cache-Reset 1 --Cache-Dir {postprocess_dir}{exo_dir}/. --Deconv-Mode SSD2 --Mask-Auto 1 --Mask-SigTh 7 --Deconv-MaxMajorIter 3 --Deconv-RMSFactor 3 --Deconv-PeakFactor 0.1 --Facets-NFacet 1 --Facets-DiamMax 5 '
        f'--Weight-OutColName DDF_WEIGHTS --GAClean-ScalesInitHMP [0] --Beam-Model NENUFAR --Beam-NBand {num_beam} --Beam-CenterNorm 1 --Beam-Smooth True  --Beam-PhasedArrayMode AE '
        f'--Freq-NBand {num_beam} --SSD2-PolyFreqOrder 3 --Freq-NDegridBand 0 --Image-NPix 1200 --Image-Cell 120 --Data-ChunkHours 0.5'
    )
    combined_ddf = f"{singularity_command} {cmd_ddf}"
    subprocess.run(combined_ddf, shell=True, check=True)

    # create a mask for SSD2 to deconvolve every source (because --Mask-Auto=1 is not as good)
    cmd_mask = (
        f'MakeMask.py --RestoredIm {postprocess_dir}{exo_dir}/Image_DI_Bis.app.restored.fits --Box 100,2 --Th 7'
    )
    combined_mask = f"{singularity_command} {cmd_mask}"
    subprocess.run(combined_mask, shell=True, check=True)

    # Continue with deconvolution, starting from the last residual (initialising model with the DicoModel generated in the previous step
    cmd_ddf = (
        f'DDF.py {postprocess_dir}{exo_dir}/Image_DI_Bis.parset --Output-Name {postprocess_dir}{exo_dir}/Image_DI_Bis.deeper --Cache-Reset 0 --Mask-Auto 0 --Mask-External {postprocess_dir}{exo_dir}/Image_DI_Bis.app.restored.fits.mask.fits '
        f'--Cache-Dirty ForceResidual --Cache-PSF Force --Predict-InitDicoModel {postprocess_dir}{exo_dir}/Image_DI_Bis.DicoModel'
    )
    combined_ddf = f"{singularity_command} {cmd_ddf}"
    subprocess.run(combined_ddf, shell=True, check=True)

    # Create a mask to remove the ATeam from the DicoModel
    cmd_mask = (
        f'MakeMask.py --RestoredIm {postprocess_dir}{exo_dir}/Image_DI_Bis.deeper.app.restored.fits --Box 100,2 --Th 10000 --ds9Mask {region_file}'
    )
    combined_mask = f"{singularity_command} {cmd_mask}"
    subprocess.run(combined_mask, shell=True, check=True)
    
    # Remove ATeam from DicoModel
    cmd_maskdico = (
        f'MaskDicoModel.py --InDicoModel {postprocess_dir}{exo_dir}/Image_DI_Bis.deeper.DicoModel --OutDicoModel {postprocess_dir}{exo_dir}/Image_DI_Bis.deeper.filterATeam.DicoModel --MaskName {postprocess_dir}{exo_dir}/Image_DI_Bis.deeper.app.restored.fits.mask.fits --InvertMask 1'
    )
    combined_maskdico = f"{singularity_command} {cmd_maskdico}"
    subprocess.run(combined_maskdico, shell=True, check=True)

    cmd_kms = (
        f'kMS.py --MSName {postprocess_dir}{exo_dir}/GSB.MS --SolverType CohJones --PolMode IFull --BaseImageName {postprocess_dir}{exo_dir}/Image_DI_Bis.deeper --dt 1 --InCol DATA --SolsDir={postprocess_dir}{exo_dir}/SOLSDIR --NodesFile Single --DDFCacheDir={postprocess_dir}{exo_dir}/ '
        f'--NChanPredictPerMS {num_beam} --NChanSols {num_beam} --NChanBeamPerMS {num_beam} --OutSolsName DD1 --UVMinMax 0.067,1000 --AppendCalSource All --FreePredictGainColName KMS_SUB:data-ATeam '
        f'--BeamModel NENUFAR --DicoModel {postprocess_dir}{exo_dir}/Image_DI_Bis.deeper.filterATeam.DicoModel --WeightInCol DDF_WEIGHTS --PhasedArrayMode AE'
    )
    combined_kms = f"{singularity_command} {cmd_kms}"
    subprocess.run(combined_kms, shell=True, check=True)




    # cmd_kms = (
    #     f'kMS.py --MSName {postprocess_dir}/{exo_dir}/GSB.MS --SolverType CohJones --PolMode IFull --BaseImageName {postprocess_dir}/{exo_dir}/GSB_Image_DI '
    #     f'--dt 6 --InCol DI_DATA --OutCol SUB_DATA --SolsDir={postprocess_dir}/{exo_dir}/SOLSDIR --NodesFile Single --DDFCacheDir={postprocess_dir}/{exo_dir}/ --NChanPredictPerMS 45 --NChanSols 45 '
    #     '--OutSolsName DD1 --UVMinMax 0.067,1000 --AppendCalSource All --FreePredictGainColName KMS_SUB:data-ATeam --BeamModel NENUFAR'
    # )



# Task 6. DynspecMS

@task(log_prints=True)
def dynspec(exo_dir: str):
    singularity_command = f"singularity exec -B/data/$USER {singularity_file}"

    # cmd_list = f'ls -d {postprocess_dir}{exo_dir}/MSB*.MS > {postprocess_dir}/{exo_dir}/mslist.txt'
    # subprocess.run(cmd_list, shell=True, check=True)

    # exo_MSB = glob.glob(postprocess_dir + exo_dir + '/MSB*.MS')
    # num_MSB = len(exo_MSB)

    cmd_ddf = (
        f'DDF.py {postprocess_dir}{exo_dir}/Image_DI_Bis.deeper.parset --Output-Name {postprocess_dir}{exo_dir}/Image_DI_Bis.subtract --Cache-Reset 1 --Cache-Dirty auto --Cache-PSF auto --Data-ColName KMS_SUB '
        f'--Weight-ColName IMAGING_WEIGHT --Predict-InitDicoModel None --Mask-External None --Mask-Auto 1 --Deconv-MaxMajorIter 3 --Output-Mode Clean --Data-MS {postprocess_dir}{exo_dir}/GSB.MS --Predict-ColName DDF_PREDICT'
    )
    combined_ddf = f"{singularity_command} {cmd_ddf}"
    subprocess.run(combined_ddf, shell=True, check=True)

    # cmd_ddf = (
    #     f'DDF.py --Data-MS {postprocess_dir}/{exo_dir}/GSB.MS --Data-ColName KMS_SUB --Output-Name {postprocess_dir}/{exo_dir}/Image_SUB --Image-Cell 60 --Image-NPix 2400 '
    #     f'--Output-Mode Clean --Facets-NFacets 5 --Parallel-NCPU 96 --Freq-NBand 45 --Freq-NDegridBand 0 --Selection-UVRangeKm [0.067,1000] '
    #     '--Comp-GridDecorr 0.0001 --Comp-DegridDecorr 0.0001 --Deconv-Mode HMP --Deconv-MaxMajorIter 20 --Mask-Auto 1 --Mask-SigTh 4 '
    #     '--Deconv-AllowNegative 0 --Deconv-RMSFactor 4 --Output-Also all --Weight-OutColName BRIGGS_WEIGHT --Output-Also all --Predict-ColName DDF_PREDICT'
    # )



    target_str = exo_dir.split("_")[4:-1]
    if len(target_str) > 1:
        target_name = "_".join(target_str)
    else:
        target_name = target_str[0]

    make_target_list(target_name, postprocess_dir, exo_dir)
    plot_target_distribution(postprocess_dir, exo_dir)

    cmd_dynspec = (
        f'ms2dynspec.py --ms {postprocess_dir}{exo_dir}/GSB.MS --data KMS_SUB --model DDF_PREDICT --rad 11 --LogBoring 1 --uv 0.067,1000 '
        f'--WeightCol IMAGING_WEIGHT --srclist {postprocess_dir}{exo_dir}/target.txt --noff 0 --NCPU 50 --TChunkHours 1 --OutDirName {postprocess_dir}{exo_dir}/dynamic_spec'
    )

    combined_dynspec = f"{singularity_command} {cmd_dynspec}"
    subprocess.run(combined_dynspec, shell=True, check=True)

# Task 7. Source-finding

@task(log_prints=True)
def source_find_v(exo_dir: str, time_windows, freq_windows):

    # get the folder name of the dynamic spectrum
    dynspec_folder = glob.glob(f'{postprocess_dir}{exo_dir}/dynamic_spec_*.MS')[0].split('/')[-1]

    # generate a MAD map to be used as a weight map in convolution
    # median_map, mad_map = generate_noise_map(f'{postprocess_dir}{exo_dir}/{dynspec_folder}/')
    mean_map, std_map = generate_noise_map_v(f'{postprocess_dir}{exo_dir}/{dynspec_folder}/')

    cmd_norm_dir = f'mkdir {postprocess_dir}{exo_dir}/{dynspec_folder}/weighted_dynamic_spec'
    subprocess.run(cmd_norm_dir, shell=True, check=True)
    generate_and_save_weight_map_v(f'{postprocess_dir}{exo_dir}/{dynspec_folder}/', f'{postprocess_dir}{exo_dir}/{dynspec_folder}/weighted_dynamic_spec/')

    # mkdir to apply the Gaussian filter
    cmd_convol_dir = f'mkdir {postprocess_dir}{exo_dir}/{dynspec_folder}/convol_gaussian/'
    subprocess.run(cmd_convol_dir, shell=True, check=True)

    # matched filtering
    dynamic_directory = f'{postprocess_dir}{exo_dir}/{dynspec_folder}/weighted_dynamic_spec/'
    convol_directory = f'{postprocess_dir}{exo_dir}/{dynspec_folder}/convol_gaussian/'

    # get the size of the dynamic spectrum, to make sure that the windows do not exceed the size
    dynspec_file = glob.glob(f'{dynamic_directory}/*.fits')[0]
    with fits.open(dynspec_file) as hdul:
        dynspec_size = hdul[0].data.shape
        time_bins = dynspec_size[1]
        freq_bins = dynspec_size[0]

    time_windows = [w for w in time_windows if w <= time_bins]
    freq_windows = [w for w in freq_windows if w <= freq_bins]

    convol_tasks = [delayed(apply_gaussian_filter)(filename, dynamic_directory, time_windows, freq_windows, convol_directory)
                       for filename in os.listdir(dynamic_directory)]
    
    compute(*convol_tasks)

    # generate noise map for the convolved dynamic spectrum
    # but we need to make a directory for the noise map first
    cmd_noise_dir = f'mkdir {postprocess_dir}{exo_dir}/{dynspec_folder}/noise_map/'
    subprocess.run(cmd_noise_dir, shell=True, check=True)

    noise_directory = f'{postprocess_dir}{exo_dir}/{dynspec_folder}/noise_map/'

    cmd_detection_dir = f'mkdir {postprocess_dir}{exo_dir}/{dynspec_folder}/detected_dynamic_spec'
    subprocess.run(cmd_detection_dir, shell=True, check=True)

    detection_directory = f'{postprocess_dir}{exo_dir}/{dynspec_folder}/detected_dynamic_spec/'

    # Not parallelized because it's opening too many files
    for t_window in time_windows:
        for f_window in freq_windows:
            calculate_noise_for_window(convol_directory, noise_directory, t_window, f_window)

    detection_tasks = [delayed(source_detection)(convol_directory, noise_directory, t_window, f_window, detection_directory, direction_threshold, direction_threshold_target, dynamic_threshold, dynamic_threshold_target)
            for t_window in time_windows
            for f_window in freq_windows]
    
    compute(*detection_tasks)

    detected_files = [f for f in glob.glob(f'{detection_directory}/*.fits') if "region" not in f.split('/')[-1]]

    detected_coor = []
    for detection in detected_files:
        filename = detection.split('/')[-1]
        source_type = filename.split('_')[0]
        source_coord = '_'.join(filename.split('_')[-2:]).replace('.fits', '')
        detected_coor.append([source_type, source_coord])

    detected_coor = np.array(detected_coor)
    detected_coor = np.unique(detected_coor, axis=0)

    for coor in detected_coor:

        sources_coor = glob.glob(f'{postprocess_dir}{exo_dir}/{dynspec_folder}/detected_dynamic_spec/{coor[0]}_*_{coor[1]}.fits')
        sources_coor.sort()

        records = []

        for source in sources_coor:
            # Extract filename
            filename = source.split('/')[-1]
            
            # Extract time and frequency from filename
            time = float(filename.split('_')[2].replace('s', ''))
            freq = float(filename.split('_')[3].replace('kHz', ''))
            
            # Open FITS file to get SNR
            with fits.open(source) as hdu:
                transient_snr = hdu[0].header['SNR']
            
            # Append the time, freq, and SNR as a dictionary to the list
            records.append({
                'source': source,
                'time': time,
                'freq': freq,
                'snr': transient_snr
            })

        # Sort the list of dictionaries by SNR
        sorted_records = sorted(records, key=lambda x: x['snr'], reverse=True)

        # Get the record with the highest SNR
        highest_snr_record = sorted_records[0]

        # Extract the time and frequency corresponding to the highest SNR
        source_with_highest_snr = highest_snr_record['source']
        time_with_highest_snr = highest_snr_record['time']
        freq_with_highest_snr = highest_snr_record['freq']

        if freq_with_highest_snr > 195 * freq_windows[0]: # and time_with_highest_snr > 8 * time_windows[0]:

            with fits.open(source_with_highest_snr) as hdu:
                snr_map = hdu[0].data
                header = hdu[0].header

                # Time axis info
                crval1 = header['CRVAL1']
                cdelt1 = header['CDELT1']
                crpix1 = header['CRPIX1']
                naxis1 = header['NAXIS1']

                # Frequency axis info
                crval2 = header['CRVAL2']
                cdelt2 = header['CDELT2']
                crpix2 = header['CRPIX2']
                naxis2 = header['NAXIS2']

                # Calculate physical values for the axes
                time_vals = crval1 + (np.arange(naxis1) - (crpix1 - 1)) * cdelt1
                freq_vals = crval2 + (np.arange(naxis2) - (crpix2 - 1)) * cdelt2

                snr_map_no_nan = np.nan_to_num(snr_map, nan=0.0)

                filename = source_with_highest_snr.split('/')[-1]

                plt.figure(figsize=(12, 4))
                plt.imshow(snr_map_no_nan, aspect='auto', origin='lower', cmap='PiYG', vmin=-7, vmax=7, extent=[time_vals[0], time_vals[-1], freq_vals[0], freq_vals[-1]])
                cbar = plt.colorbar(shrink=0.95, aspect=15, pad=0.02)

                # Add a label to the colorbar and bring it closer
                cbar.set_label('SNR', rotation=270, labelpad=10)
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (MHz)')
                plt.title(f'SNR Map for {filename}')

                plt.savefig(f'{detection_directory}/{filename}.png', dpi=200, bbox_inches='tight')
                plt.close()

    # Make a directory
    cmd_png_dir = f'mkdir {postprocess_dir}{exo_dir}/{dynspec_folder}/{exo_dir}_png/'
    subprocess.run(cmd_png_dir, shell=True, check=True)

    png_files = glob.glob(f'{detection_directory}/*.png')
    if png_files:
        # Only run the command if there are .png files
        cmd_mv_png = f'mv {detection_directory}/*.png {postprocess_dir}{exo_dir}/{dynspec_folder}/{exo_dir}_png/'
        subprocess.run(cmd_mv_png, shell=True, check=True)
    else:
        print("No .png files found in the directory.")

    # Move the png files to the directory
    # cmd_mv_png = f'mv {detection_directory}/*.png {postprocess_dir}{exo_dir}/{dynspec_folder}/{exo_dir}_png/'
    # subprocess.run(cmd_mv_png, shell=True, check=True)

    # seventh, remove some directories within dynamic_spec
    cmd_remo_dyna = f"rm -rf {postprocess_dir}/{exo_dir}/{dynspec_folder}/convol_gaussian {postprocess_dir}/{exo_dir}/{dynspec_folder}/noise_map" #{postprocess_dir}/{exo_dir}/{dynspec_folder}/weighted_dynamic_spec"
    subprocess.run(cmd_remo_dyna, shell=True, check=True)

    cmd_rename = f"mv {postprocess_dir}/{exo_dir}/{dynspec_folder}/detected_dynamic_spec {postprocess_dir}/{exo_dir}/{dynspec_folder}/detected_dynamic_spec_v"
    subprocess.run(cmd_rename, shell=True, check=True)

    cmd_rename = f"mv {postprocess_dir}/{exo_dir}/{dynspec_folder}/weighted_dynamic_spec {postprocess_dir}/{exo_dir}/{dynspec_folder}/weighted_dynamic_spec_v"
    subprocess.run(cmd_rename, shell=True, check=True)

    cmd_rename = f"mv {postprocess_dir}/{exo_dir}/{dynspec_folder}/{exo_dir}_png {postprocess_dir}/{exo_dir}/{dynspec_folder}/{exo_dir}_png_v"
    subprocess.run(cmd_rename, shell=True, check=True)

@task(log_prints=True)
def source_find_i(exo_dir: str, time_windows, freq_windows):

    # get the folder name of the dynamic spectrum
    dynspec_folder = glob.glob(f'{postprocess_dir}{exo_dir}/dynamic_spec_*.MS')[0].split('/')[-1]

    # generate a MAD map to be used as a weight map in convolution
    # median_map, mad_map = generate_noise_map(f'{postprocess_dir}{exo_dir}/{dynspec_folder}/')
    mean_map, std_map = generate_noise_map_i(f'{postprocess_dir}{exo_dir}/{dynspec_folder}/')

    cmd_norm_dir = f'mkdir {postprocess_dir}{exo_dir}/{dynspec_folder}/weighted_dynamic_spec'
    subprocess.run(cmd_norm_dir, shell=True, check=True)
    generate_and_save_weight_map_i(f'{postprocess_dir}{exo_dir}/{dynspec_folder}/', f'{postprocess_dir}{exo_dir}/{dynspec_folder}/weighted_dynamic_spec/')

    # mkdir to apply the Gaussian filter
    cmd_convol_dir = f'mkdir {postprocess_dir}{exo_dir}/{dynspec_folder}/convol_gaussian/'
    subprocess.run(cmd_convol_dir, shell=True, check=True)

    # matched filtering
    dynamic_directory = f'{postprocess_dir}{exo_dir}/{dynspec_folder}/weighted_dynamic_spec/'
    convol_directory = f'{postprocess_dir}{exo_dir}/{dynspec_folder}/convol_gaussian/'

    # get the size of the dynamic spectrum, to make sure that the windows do not exceed the size
    dynspec_file = glob.glob(f'{dynamic_directory}/*.fits')[0]
    with fits.open(dynspec_file) as hdul:
        dynspec_size = hdul[0].data.shape
        time_bins = dynspec_size[1]
        freq_bins = dynspec_size[0]

    time_windows = [w for w in time_windows if w <= time_bins]
    freq_windows = [w for w in freq_windows if w <= freq_bins]

    convol_tasks = [delayed(apply_gaussian_filter)(filename, dynamic_directory, time_windows, freq_windows, convol_directory)
                       for filename in os.listdir(dynamic_directory)]
    
    compute(*convol_tasks)

    # generate noise map for the convolved dynamic spectrum
    # but we need to make a directory for the noise map first
    cmd_noise_dir = f'mkdir {postprocess_dir}{exo_dir}/{dynspec_folder}/noise_map/'
    subprocess.run(cmd_noise_dir, shell=True, check=True)

    noise_directory = f'{postprocess_dir}{exo_dir}/{dynspec_folder}/noise_map/'

    cmd_detection_dir = f'mkdir {postprocess_dir}{exo_dir}/{dynspec_folder}/detected_dynamic_spec'
    subprocess.run(cmd_detection_dir, shell=True, check=True)

    detection_directory = f'{postprocess_dir}{exo_dir}/{dynspec_folder}/detected_dynamic_spec/'

    # Not parallelized because it's opening too many files
    for t_window in time_windows:
        for f_window in freq_windows:
            calculate_noise_for_window(convol_directory, noise_directory, t_window, f_window)

    detection_tasks = [delayed(source_detection)(convol_directory, noise_directory, t_window, f_window, detection_directory, direction_threshold, direction_threshold_target, dynamic_threshold, dynamic_threshold_target)
            for t_window in time_windows
            for f_window in freq_windows]
    
    compute(*detection_tasks)

    detected_files = [f for f in glob.glob(f'{detection_directory}/*.fits') if "region" not in f.split('/')[-1]]

    detected_coor = []
    for detection in detected_files:
        filename = detection.split('/')[-1]
        source_type = filename.split('_')[0]
        source_coord = '_'.join(filename.split('_')[-2:]).replace('.fits', '')
        detected_coor.append([source_type, source_coord])

    detected_coor = np.array(detected_coor)
    detected_coor = np.unique(detected_coor, axis=0)

    for coor in detected_coor:

        sources_coor = glob.glob(f'{postprocess_dir}{exo_dir}/{dynspec_folder}/detected_dynamic_spec/{coor[0]}_*_{coor[1]}.fits')
        sources_coor.sort()

        records = []

        for source in sources_coor:
            # Extract filename
            filename = source.split('/')[-1]
            
            # Extract time and frequency from filename
            time = float(filename.split('_')[2].replace('s', ''))
            freq = float(filename.split('_')[3].replace('kHz', ''))
            
            # Open FITS file to get SNR
            with fits.open(source) as hdu:
                transient_snr = hdu[0].header['SNR']
            
            # Append the time, freq, and SNR as a dictionary to the list
            records.append({
                'source': source,
                'time': time,
                'freq': freq,
                'snr': transient_snr
            })

        # Sort the list of dictionaries by SNR
        sorted_records = sorted(records, key=lambda x: x['snr'], reverse=True)

        # Get the record with the highest SNR
        highest_snr_record = sorted_records[0]

        # Extract the time and frequency corresponding to the highest SNR
        source_with_highest_snr = highest_snr_record['source']
        time_with_highest_snr = highest_snr_record['time']
        freq_with_highest_snr = highest_snr_record['freq']

        if freq_with_highest_snr > 195 * freq_windows[0]: # and time_with_highest_snr > 8 * time_windows[0]:

            with fits.open(source_with_highest_snr) as hdu:
                snr_map = hdu[0].data
                header = hdu[0].header

                # Time axis info
                crval1 = header['CRVAL1']
                cdelt1 = header['CDELT1']
                crpix1 = header['CRPIX1']
                naxis1 = header['NAXIS1']

                # Frequency axis info
                crval2 = header['CRVAL2']
                cdelt2 = header['CDELT2']
                crpix2 = header['CRPIX2']
                naxis2 = header['NAXIS2']

                # Calculate physical values for the axes
                time_vals = crval1 + (np.arange(naxis1) - (crpix1 - 1)) * cdelt1
                freq_vals = crval2 + (np.arange(naxis2) - (crpix2 - 1)) * cdelt2

                snr_map_no_nan = np.nan_to_num(snr_map, nan=0.0)

                filename = source_with_highest_snr.split('/')[-1]

                plt.figure(figsize=(12, 4))
                plt.imshow(snr_map_no_nan, aspect='auto', origin='lower', cmap='PiYG', vmin=-7, vmax=7, extent=[time_vals[0], time_vals[-1], freq_vals[0], freq_vals[-1]])
                cbar = plt.colorbar(shrink=0.95, aspect=15, pad=0.02)

                # Add a label to the colorbar and bring it closer
                cbar.set_label('SNR', rotation=270, labelpad=10)
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (MHz)')
                plt.title(f'SNR Map for {filename}')

                plt.savefig(f'{detection_directory}/{filename}.png', dpi=200, bbox_inches='tight')
                plt.close()

    # Make a directory
    cmd_png_dir = f'mkdir {postprocess_dir}{exo_dir}/{dynspec_folder}/{exo_dir}_png/'
    subprocess.run(cmd_png_dir, shell=True, check=True)

    png_files = glob.glob(f'{detection_directory}/*.png')
    if png_files:
        # Only run the command if there are .png files
        cmd_mv_png = f'mv {detection_directory}/*.png {postprocess_dir}{exo_dir}/{dynspec_folder}/{exo_dir}_png/'
        subprocess.run(cmd_mv_png, shell=True, check=True)
    else:
        print("No .png files found in the directory.")

    # Move the png files to the directory
    # cmd_mv_png = f'mv {detection_directory}/*.png {postprocess_dir}{exo_dir}/{dynspec_folder}/{exo_dir}_png/'
    # subprocess.run(cmd_mv_png, shell=True, check=True)

    # seventh, remove some directories within dynamic_spec
    cmd_remo_dyna = f"rm -rf {postprocess_dir}/{exo_dir}/{dynspec_folder}/convol_gaussian {postprocess_dir}/{exo_dir}/{dynspec_folder}/noise_map" #{postprocess_dir}/{exo_dir}/{dynspec_folder}/weighted_dynamic_spec"
    subprocess.run(cmd_remo_dyna, shell=True, check=True)

    cmd_rename = f"mv {postprocess_dir}/{exo_dir}/{dynspec_folder}/detected_dynamic_spec {postprocess_dir}/{exo_dir}/{dynspec_folder}/detected_dynamic_spec_i"
    subprocess.run(cmd_rename, shell=True, check=True)

    cmd_rename = f"mv {postprocess_dir}/{exo_dir}/{dynspec_folder}/weighted_dynamic_spec {postprocess_dir}/{exo_dir}/{dynspec_folder}/weighted_dynamic_spec_i"
    subprocess.run(cmd_rename, shell=True, check=True)

    cmd_rename = f"mv {postprocess_dir}/{exo_dir}/{dynspec_folder}/{exo_dir}_png {postprocess_dir}/{exo_dir}/{dynspec_folder}/{exo_dir}_png_i"
    subprocess.run(cmd_rename, shell=True, check=True)


# Task 8. Clear up the directory

@task(log_prints=True)
def clearup(exo_dir: str):

    # first, remove all the SB*.MS files
    cmd_remo_SB = f"rm -rf {postprocess_dir}/{exo_dir}/SB???.MS"
    subprocess.run(cmd_remo_SB, shell=True, check=True)

    # second, make a directory to keep some images
    cmd_image_dir = f'mkdir {postprocess_dir}/{exo_dir}/archive_images/'
    subprocess.run(cmd_image_dir, shell=True, check=True)

    # # third, move the images to the directory
    # cmd_mv_image = f'mv {postprocess_dir}/{exo_dir}/MSB??_Image_SUB.dirty*.fits {postprocess_dir}/{exo_dir}/archive_images/' 
    # subprocess.run(cmd_mv_image, shell=True, check=True)

    # fourth, move some other images to the directory
    cmd_mv_image = f'mv {postprocess_dir}/{exo_dir}/Image_DI_Bis.subtract.app.*.fits {postprocess_dir}/{exo_dir}/archive_images/'
    subprocess.run(cmd_mv_image, shell=True, check=True)

    # fifth, remove the MSB files and Image_SUB files
    cmd_remo_MSB = f"rm -rf {postprocess_dir}/{exo_dir}/MSB* {postprocess_dir}/{exo_dir}/GSB* {postprocess_dir}/{exo_dir}/Image*"
    subprocess.run(cmd_remo_MSB, shell=True, check=True)

    # sixth, remove the other files
    cmd_remo_other = f"rm -rf {postprocess_dir}/{exo_dir}/SOLSDIR {postprocess_dir}/{exo_dir}/dynamic_spec_DynSpecs_*.tgz"
    subprocess.run(cmd_remo_other, shell=True, check=True)

###### Here come the flows (functions calling the tasks) #######

@flow(name="EXO_IMG PIPELINE", log_prints=True)
def exo_pipe(exo_dir):
    with open(lockfile, "w") as f:
        f.write("Processing ongoing")

    task_copy_calibrator = copy_calibrator_data.submit(exo_dir)

    task_copy_target = copy_target_data.submit(exo_dir)

    # Has calibrator been processed already? If not, find bad MA and do A-team calibration
    cal, cal_dir, cali_check = task_copy_calibrator.result()

    if cali_check == False:
        bad_MAs = identify_bad_mini_arrays(cal, cal_dir)
        calibration_Ateam(cal, cal_dir, bad_MAs)
    else:
        with open(f'{postprocess_dir}/{cal_dir}/bad_MA.txt', 'r') as bad_MA_text:
            bad_MAs = bad_MA_text.read().strip()
        # No need to do Ateam calibration if it's done previously

    apply_Ateam_solution(cal_dir, exo_dir, bad_MAs, wait_for=[task_copy_calibrator, task_copy_target])

    subtract_Ateam(exo_dir)

    dynspec(exo_dir)

    source_find_v(exo_dir, time_windows, freq_windows)

    source_find_i(exo_dir, time_windows, freq_windows)

    clearup(exo_dir)

    os.remove(lockfile)

@flow(name='Check Flow', log_prints=True)
def check_flow():
    if os.path.exists(lockfile):
        print("Exiting due to existence of lockfile")
        return Completed(message="Lockfile exists, skipping run.")

    new_data = check_new_data(watch_dir, postprocess_dir)

    if len(new_data) > 0:
        # Trigger the main flow
        for unprocessed_data in new_data:
            exo_pipe(unprocessed_data)

        # exo_pipe(new_data[0])

    return Completed(message="Run completed without issues.")

if __name__ == "__main__":
    check = check_flow.serve(name="check-flow", interval=21600)