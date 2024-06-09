#!/usr/bin/env python3

import subprocess
import os, glob
from astropy.io import fits
import numpy as np
from dask import delayed, compute
from casatools import table
from templates.Find_Bad_MAs_template import find_bad_MAs_multi
from templates.Make_Target_List_template import make_target_list_FRB
from templates.Plot_target_distri_template import plot_target_distribution_FRB
from templates.Noise_esti_template import generate_noise_map_v, calculate_noise_for_window, apply_gaussian_filter
from templates.Noise_esti_template import generate_and_save_weight_map_v, source_detection, generate_noise_map_i, generate_and_save_weight_map_i
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

###### Initial settings ######

# Set file locations
# watch_dir = "/databf/nenufar-nri/LT02/202?/??/*HD_189733*"

# preprocess_dir = "/databf/nenufar-nri/LT02/"
postprocess_dir = "/data/xzhang/exo_img/"
pipe_dir = "/home/xzhang/software/exo_img_pipe/"
# lockfile = "/home/xzhang/software/exo_img_pipe/lock.file"
singularity_file = "/home/xzhang/software/ddf_dev2_ateam.sif"
# skip_file = "/home/xzhang/software/exo_img_pipe/templates/skip.txt"

# Calibrators
CALIBRATORS = ['CYG_A', 'CAS_A', 'TAU_A', 'VIR_A']

# How many SB per processing chunk
chunk_num = 30

cal = 'CYG_A'
cali_check = True
cal_dir = '20240607_020000_20240607_022000_CYG_A_TRACKING/L1'
exo_dir = '20240607_022000_20240607_050000_FRB20240114A_TRACKING/L1'
target_name = 'FRB20240114A'

# How many channels per SB
chan_per_SB_origin = 6
ave_chan = 1
chan_per_SB = int(chan_per_SB_origin/ave_chan)

# chan_per_SB = 12

# Avoid bad channel making KMS hang
# bin_per_MSB = chunk_num // 3

# the lowest SB we use
SB_min = 106 # 92
SB_ave_kms = 5

# The region file we use for A-team removal
region_file = "/home/xzhang/software/exo_img_pipe/regions/Ateam.reg"

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



###### Here are the tasks (aka functions doing the job) ######

def identify_bad_mini_arrays(cal: str, cal_dir: str) -> str:
    # Step 1: Set the environment
    # cmd = "use DP3"
    # subprocess.run(cmd, shell=True, check=True)

    # Step 2: Run DP3 DPPP-aoflagger.parset command
    cali_SB_0 = glob.glob(postprocess_dir + cal_dir + '/SB*.MS')
    cali_SB = [f for f in cali_SB_0 if int(f.split('/SB')[1].split('.MS')[0]) > SB_min]
    cali_SB.sort()

    # Now we need to make sure that the number of SB is a multiple of SB_ave_kms. We can remove the first few SBs if necessary
    # When we process data with high reso (large size), we use chunk_num rather than SB_ave_kms
    num_SB = len(cali_SB)
    # num_remove = num_SB % SB_ave_kms
    num_remove = num_SB % chunk_num
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
    num_chunks = len(cali_SB) // chunk_num

    cmd_sky_dir = f"mkdir {postprocess_dir}/{cal_dir}/sky_models/"
    subprocess.run(cmd_sky_dir, shell=True, check=True)

    for i in range(num_chunks):
        # Extract the ith chunk of chunk_num file names
        chunk = cali_SB[i * chunk_num: (i + 1) * chunk_num]

        # Create the msin string by joining the chunk with commas
        SB_str = ",".join(chunk)

        # Construct the output file name using the loop index (i+1)
        MSB_filename = f"{postprocess_dir}/{cal_dir}/MSB{str(i).zfill(2)}.MS"

        # Construct the command string with the msin argument and the msout argument
        cmd_flagchan = f"DP3 {pipe_dir}/templates/DPPP-flagchan.parset msin=[{SB_str}] msout={MSB_filename} avg.freqstep={ave_chan}"
        subprocess.run(cmd_flagchan, shell=True, check=True)

        # Construct the command string with the msin argument and the msout argument
        cmd_aoflagger = f"DP3 {pipe_dir}/templates/DPPP-aoflagger.parset msin={MSB_filename} flag.strategy={pipe_dir}/templates/Nenufar64C1S_FRB.lua"
        subprocess.run(cmd_aoflagger, shell=True, check=True)

        cmd_cali = f"calpipe {postprocess_dir}/{cal_dir}/cali.toml {MSB_filename}"
        subprocess.run(cmd_cali, shell=True, check=True)

        cmd_makesky = f"mkdir {postprocess_dir}/{cal_dir}/sky_models/MSB{str(i).zfill(2)}/"
        subprocess.run(cmd_makesky, shell=True, check=True)

        cmd_movesky = f"mv {MSB_filename}/sky_model {postprocess_dir}/{cal_dir}/sky_models/MSB{str(i).zfill(2)}/"
        subprocess.run(cmd_movesky, shell=True, check=True)

    # Step 3: Call the imported function directly
    bad_MAs = find_bad_MAs_multi(f"{postprocess_dir}/{cal_dir}/")

    # Step 4: Remove the testing MSB
    cmd_remo_MSB = f"rm -rf {postprocess_dir}/{cal_dir}/MSB*.MS"
    subprocess.run(cmd_remo_MSB, shell=True, check=True)

    return bad_MAs

# Task 3. Calibration with A team

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
    num_remove = num_SB % chunk_num
    cali_SB = cali_SB[num_remove:]

    # Use casatools to get the list of antennas observed
    tb = table()
    tb.open(cali_SB[0] + '/ANTENNA')
    antennas = tb.getcol('NAME')
    tb.close()

    # save the list of antennas to a file, seperated by commas
    with open(f'{postprocess_dir}/{cal_dir}/All_MAs.txt', 'w') as f:
        f.write(','.join(antennas))

    # remove the bad MAs rather than flagging them
    with open(f'{pipe_dir}/templates/DPPP-removeant.parset', 'r') as template_remove:
        remove_content = template_remove.read()

    modified_remove_content = remove_content.replace('MA_TO_REMOVE', bad_MAs)

    # Write the modified content to a new file
    with open(f'{postprocess_dir}/{cal_dir}/DPPP-removeant.parset', 'w') as remove_file:
        remove_file.write(modified_remove_content)

    # Determine the number of full chunks of chunk_num we can form
    num_chunks = len(cali_SB) // chunk_num

    for i in range(num_chunks):
        # Extract the ith chunk of chunk_num file names
        chunk = cali_SB[i * chunk_num: (i + 1) * chunk_num]

        # Create the msin string by joining the chunk with commas
        SB_str = ",".join(chunk)

        # Construct the output file name using the loop index (i+1)
        MSB_filename = f"{postprocess_dir}/{cal_dir}/MSB{str(i).zfill(2)}.MS"

        # Construct the command string with the msin argument and the msout argument
        cmd_flagchan = f"DP3 {pipe_dir}/templates/DPPP-flagchan.parset msin=[{SB_str}] msout={MSB_filename} avg.freqstep={ave_chan}"
        subprocess.run(cmd_flagchan, shell=True, check=True)

        # We need to remove the bad MA rather than flagging them
        cmd_removeMA = f"DP3 {postprocess_dir}/{cal_dir}/DPPP-removeant.parset msin={MSB_filename} msout={postprocess_dir}/{cal_dir}/MSB{str(i).zfill(2)}.MSB"
        subprocess.run(cmd_removeMA, shell=True, check=True)
        # remove original MSB
        cmd_remo_MSB = f"rm -rf {MSB_filename}"
        subprocess.run(cmd_remo_MSB, shell=True, check=True)
        # rename the new MSB
        cmd_rename_MSB = f"mv {postprocess_dir}/{cal_dir}/MSB{str(i).zfill(2)}.MSB {MSB_filename}"
        subprocess.run(cmd_rename_MSB, shell=True, check=True)

        # Construct the command string with the msin argument and the msout argument
        cmd_aoflagger = f"DP3 {pipe_dir}/templates/DPPP-aoflagger.parset msin={MSB_filename} flag.strategy={pipe_dir}/templates/Nenufar64C1S_FRB.lua"
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

        cmd_copysky = f"cp -rf {postprocess_dir}/{cal_dir}/sky_models/MSB{str(i).zfill(2)}/sky_model {MSB_filename}/"
        subprocess.run(cmd_copysky, shell=True, check=True)

        cmd_cali = f"calpipe {postprocess_dir}/{cal_dir}/cali.toml {MSB_filename}"
        subprocess.run(cmd_cali, shell=True, check=True)

        # Remove the table files so they don't take up too much space!
        cmd_remo_table = f"rm -rf {MSB_filename}/table.* {MSB_filename}/pre_cal_flags.h5"
        subprocess.run(cmd_remo_table, shell=True, check=True)

# Task 4. Apply A-team calibration solution to target

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
    num_remove = num_SB % chunk_num
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

    # Determine the number of full chunks of chunk_num we can form
    num_chunks = len(exo_SB) // chunk_num

    for i in range(num_chunks):
        # Extract the ith chunk of chunk_num file names
        chunk = exo_SB[i * chunk_num: (i + 1) * chunk_num]

        # Create the msin string by joining the chunk with commas
        SB_str = ",".join(chunk)

        # Construct the output file name using the loop index (i+1)
        MSB_filename = f"{postprocess_dir}/{exo_dir}/MSB{str(i).zfill(2)}.MS"        # Extract the ith chunk of chunk_num file names

        # Construct the command string with the msin argument and the msout argument
        cmd_flagchan = f"DP3 {pipe_dir}/templates/DPPP-flagchan.parset msin=[{SB_str}] msout={MSB_filename} avg.freqstep={ave_chan}"
        subprocess.run(cmd_flagchan, shell=True, check=True)

        # only run removeant when there are antennas to be removed
        if len(remove_antennas) > 0:
            cmd_removeMA = f"DP3 {postprocess_dir}/{exo_dir}/DPPP-removeant.parset msin={MSB_filename} msout={MSB_filename}B"
            subprocess.run(cmd_removeMA, shell=True, check=True)
            # remove original MSB
            cmd_remo_MSB = f"rm -rf {MSB_filename}"
            subprocess.run(cmd_remo_MSB, shell=True, check=True)
            # rename the new MSB
            cmd_rename_MSB = f"mv {MSB_filename}B {MSB_filename}"
            subprocess.run(cmd_rename_MSB, shell=True, check=True)

        # Construct the command string with the msin argument and the msout argument
        cmd_aoflagger = f"DP3 {pipe_dir}/templates/DPPP-aoflagger.parset msin={MSB_filename} flag.strategy={pipe_dir}/templates/Nenufar64C1S_FRB.lua"
        subprocess.run(cmd_aoflagger, shell=True, check=True)

        # Copy calibration solution
        cmd_copy_solution = f'cp {postprocess_dir}/{cal_dir}/MSB{str(i).zfill(2)}.MS/instrument_ddecal.h5 {MSB_filename}/instrument_dical.h5'
        subprocess.run(cmd_copy_solution, shell=True, check=True)

        # apply solution
        cmd_apply_solution = f'calpipe {pipe_dir}/templates/cali_tran.toml {MSB_filename}'
        subprocess.run(cmd_apply_solution, shell=True, check=True)

        # second round of aoflagger
        cmd_aoflagger = f"DP3 {pipe_dir}/templates/DPPP-aoflagger.parset msin={MSB_filename} msin.datacolumn=DI_DATA flag.strategy={pipe_dir}/templates/Nenufar64C1S_FRB.lua"
        subprocess.run(cmd_aoflagger, shell=True, check=True)

    # remove the SB files
    cmd_remo_SB = f"rm -rf {postprocess_dir}/{exo_dir}/SB*.MS"
    subprocess.run(cmd_remo_SB, shell=True, check=True)


# Task 5. Subtract A-team from field

def subtract_Ateam(exo_dir: str):
    # Step 1: Set the environment
    singularity_command = f"singularity exec -B/data/$USER {singularity_file}"

    exo_MSB = glob.glob(postprocess_dir + exo_dir + '/MSB*.MS')
    exo_MSB.sort()
    num_MSB = len(exo_MSB)

    # we need the number of beams for the following steps
    num_beam = int(chunk_num / SB_ave_kms)

    for i in range(num_MSB):
        # create a ddf command without beam model for a test
        cmd_ddf = (
            f'DDF.py {pipe_dir}/templates/template_DI.parset --Data-MS {exo_MSB[i]} --Data-ColName DI_DATA --Output-Name {postprocess_dir}{exo_dir}/MSB{str(i).zfill(2)}_Image_DI_Bis '
            f'--Cache-Reset 1 --Cache-Dir {postprocess_dir}{exo_dir}/. --Deconv-Mode SSD2 --Mask-Auto 1 --Mask-SigTh 7 --Deconv-MaxMajorIter 3 --Deconv-RMSFactor 1 --Deconv-PeakFactor 0.1 --Facets-NFacet 1 --Facets-DiamMax 5 '
            f'--Weight-OutColName DDF_WEIGHTS --GAClean-ScalesInitHMP [0] --Beam-Model None '
            f'--Freq-NBand {num_beam} --SSD2-PolyFreqOrder 2 --Freq-NDegridBand 0 --Image-NPix 1200 --Image-Cell 120 --Data-ChunkHours 0.5'
        )
        combined_ddf = f"{singularity_command} {cmd_ddf}"
        subprocess.run(combined_ddf, shell=True, check=True)

        # create a mask for SSD2 to deconvolve every source (because --Mask-Auto=1 is not as good)
        cmd_mask = (
            f'MakeMask.py --RestoredIm {postprocess_dir}{exo_dir}/MSB{str(i).zfill(2)}_Image_DI_Bis.app.restored.fits --Box 100,2 --Th 7'
        )
        combined_mask = f"{singularity_command} {cmd_mask}"
        subprocess.run(combined_mask, shell=True, check=True)

        # Continue with deconvolution, starting from the last residual (initialising model with the DicoModel generated in the previous step
        cmd_ddf = (
            f'DDF.py {postprocess_dir}{exo_dir}/MSB{str(i).zfill(2)}_Image_DI_Bis.parset --Output-Name {postprocess_dir}{exo_dir}/MSB{str(i).zfill(2)}_Image_DI_Bis.deeper --Cache-Reset 0 --Mask-Auto 0 --Mask-External {postprocess_dir}{exo_dir}/MSB{str(i).zfill(2)}_Image_DI_Bis.app.restored.fits.mask.fits '
            f'--Cache-Dirty ForceResidual --Cache-PSF Force --Predict-InitDicoModel {postprocess_dir}{exo_dir}/MSB{str(i).zfill(2)}_Image_DI_Bis.DicoModel'
        )
        combined_ddf = f"{singularity_command} {cmd_ddf}"
        subprocess.run(combined_ddf, shell=True, check=True)

        # Create a mask to remove the ATeam from the DicoModel
        cmd_mask = (
            f'MakeMask.py --RestoredIm {postprocess_dir}{exo_dir}/MSB{str(i).zfill(2)}_Image_DI_Bis.deeper.app.restored.fits --Box 100,2 --Th 10000 --ds9Mask {region_file}'
        )
        combined_mask = f"{singularity_command} {cmd_mask}"
        subprocess.run(combined_mask, shell=True, check=True)
    
        # Remove ATeam from DicoModel
        cmd_maskdico = (
            f'MaskDicoModel.py --InDicoModel {postprocess_dir}{exo_dir}/MSB{str(i).zfill(2)}_Image_DI_Bis.deeper.DicoModel --OutDicoModel {postprocess_dir}{exo_dir}/MSB{str(i).zfill(2)}_Image_DI_Bis.deeper.filterATeam.DicoModel --MaskName {postprocess_dir}{exo_dir}/MSB{str(i).zfill(2)}_Image_DI_Bis.deeper.app.restored.fits.mask.fits --InvertMask 1'
        )
        combined_maskdico = f"{singularity_command} {cmd_maskdico}"
        subprocess.run(combined_maskdico, shell=True, check=True)

        # kms without beam model
        cmd_kms = (
            f'kMS.py --MSName {exo_MSB[i]} --SolverType CohJones --PolMode IFull --BaseImageName {postprocess_dir}{exo_dir}/MSB{str(i).zfill(2)}_Image_DI_Bis.deeper --dt 1 --InCol DI_DATA --SolsDir={postprocess_dir}{exo_dir}/SOLSDIR --NodesFile Single --DDFCacheDir={postprocess_dir}{exo_dir}/ '
            f'--NChanPredictPerMS {num_beam} --NChanSols {num_beam} --OutSolsName DD1 --UVMinMax 0.067,1000 --AppendCalSource All --FreePredictGainColName KMS_SUB:data-ATeam '
            f'--DicoModel {postprocess_dir}{exo_dir}/MSB{str(i).zfill(2)}_Image_DI_Bis.deeper.filterATeam.DicoModel --WeightInCol DDF_WEIGHTS'
        )
        combined_kms = f"{singularity_command} {cmd_kms}"
        subprocess.run(combined_kms, shell=True, check=True)

        # Remove the .ddfcache directory to save space
        cmd_remo_cache = f"rm -rf {postprocess_dir}{exo_dir}/*.ddfcache"
        subprocess.run(cmd_remo_cache, shell=True, check=True)

# Task 6. DynspecMS

def dynspec(exo_dir: str):
    singularity_command = f"singularity exec -B/data/$USER {singularity_file}"

    cmd_list = f'ls -d {postprocess_dir}{exo_dir}/MSB*.MS > {postprocess_dir}/{exo_dir}/mslist.txt'
    subprocess.run(cmd_list, shell=True, check=True)

    exo_MSB = glob.glob(postprocess_dir + exo_dir + '/MSB*.MS')
    num_MSB = len(exo_MSB)

    cmd_ddf = (
        f'DDF.py {postprocess_dir}{exo_dir}/MSB00_Image_DI_Bis.deeper.parset --Output-Name {postprocess_dir}{exo_dir}/Image_DI_Bis.subtract --Cache-Reset 1 --Cache-Dirty auto --Cache-PSF auto --Data-ColName KMS_SUB --Freq-NBand {num_MSB} '
        f'--Weight-ColName IMAGING_WEIGHT --SSD2-PolyFreqOrder 3 --Predict-InitDicoModel None --Mask-External None --Mask-Auto 1 --Deconv-MaxMajorIter 3 --Output-Mode Clean --Data-MS {postprocess_dir}{exo_dir}/mslist.txt --Predict-ColName DDF_PREDICT'
    )
    combined_ddf = f"{singularity_command} {cmd_ddf}"
    subprocess.run(combined_ddf, shell=True, check=True)

    make_target_list_FRB(target_name, postprocess_dir, exo_dir)
    plot_target_distribution_FRB(postprocess_dir, exo_dir)

    # target_file = f'{postprocess_dir}{exo_dir}/target.txt'

    # with open(target_file, 'r') as file:
    #     lines = file.readlines()

    # first_line = lines[0]

    # with open(target_file, 'w') as file:
    #     file.write(first_line)

    cmd_dynspec = (
        f'ms2dynspec.py --ms {postprocess_dir}{exo_dir}/mslist.txt --data KMS_SUB --model DDF_PREDICT --rad 11 --LogBoring 1 --uv 0.067,1000 '
        f'--WeightCol IMAGING_WEIGHT --srclist {postprocess_dir}{exo_dir}/target.txt --noff 0 --NCPU 96 --TChunkHours 0.5 --OutDirName {postprocess_dir}{exo_dir}/dynamic_spec'
    )

    combined_dynspec = f"{singularity_command} {cmd_dynspec}"
    subprocess.run(combined_dynspec, shell=True, check=True)

# Task 7. Source-finding

def source_find_v(exo_dir: str, time_windows, freq_windows):

    # get the folder name of the dynamic spectrum
    dynspec_folder = glob.glob(f'{postprocess_dir}{exo_dir}/dynamic_spec_*.MS')[0].split('/')[-1]

    # generate a MAD map to be used as a weight map in convolution
    # median_map, mad_map = generate_noise_map(f'{postprocess_dir}{exo_dir}/{dynspec_folder}/')
    mean_map, std_map = generate_noise_map_v(f'{postprocess_dir}{exo_dir}/{dynspec_folder}/')

    cmd_norm_dir = f'mkdir {postprocess_dir}{exo_dir}/{dynspec_folder}/weighted_dynamic_spec'
    subprocess.run(cmd_norm_dir, shell=True, check=True)
    generate_and_save_weight_map_v(f'{postprocess_dir}{exo_dir}/{dynspec_folder}/', f'{postprocess_dir}{exo_dir}/{dynspec_folder}/weighted_dynamic_spec/')

    # # mkdir to apply the Gaussian filter
    # cmd_convol_dir = f'mkdir {postprocess_dir}{exo_dir}/{dynspec_folder}/convol_gaussian/'
    # subprocess.run(cmd_convol_dir, shell=True, check=True)

    # # matched filtering
    # dynamic_directory = f'{postprocess_dir}{exo_dir}/{dynspec_folder}/weighted_dynamic_spec/'
    # convol_directory = f'{postprocess_dir}{exo_dir}/{dynspec_folder}/convol_gaussian/'

    # # get the size of the dynamic spectrum, to make sure that the windows do not exceed the size
    # dynspec_file = glob.glob(f'{dynamic_directory}/*.fits')[0]
    # with fits.open(dynspec_file) as hdul:
    #     dynspec_size = hdul[0].data.shape
    #     time_bins = dynspec_size[1]
    #     freq_bins = dynspec_size[0]

    # time_windows = [w for w in time_windows if w <= time_bins]
    # freq_windows = [w for w in freq_windows if w <= freq_bins]

    # convol_tasks = [delayed(apply_gaussian_filter)(filename, dynamic_directory, time_windows, freq_windows, convol_directory)
    #                    for filename in os.listdir(dynamic_directory)]
    
    # compute(*convol_tasks)

    # # generate noise map for the convolved dynamic spectrum
    # # but we need to make a directory for the noise map first
    # cmd_noise_dir = f'mkdir {postprocess_dir}{exo_dir}/{dynspec_folder}/noise_map/'
    # subprocess.run(cmd_noise_dir, shell=True, check=True)

    # noise_directory = f'{postprocess_dir}{exo_dir}/{dynspec_folder}/noise_map/'

    # cmd_detection_dir = f'mkdir {postprocess_dir}{exo_dir}/{dynspec_folder}/detected_dynamic_spec'
    # subprocess.run(cmd_detection_dir, shell=True, check=True)

    # detection_directory = f'{postprocess_dir}{exo_dir}/{dynspec_folder}/detected_dynamic_spec/'

    # # Not parallelized because it's opening too many files
    # for t_window in time_windows:
    #     for f_window in freq_windows:
    #         calculate_noise_for_window(convol_directory, noise_directory, t_window, f_window)

    # detection_tasks = [delayed(source_detection)(convol_directory, noise_directory, t_window, f_window, detection_directory, direction_threshold, direction_threshold_target, dynamic_threshold, dynamic_threshold_target)
    #         for t_window in time_windows
    #         for f_window in freq_windows]
    
    # compute(*detection_tasks)

    # detected_files = [f for f in glob.glob(f'{detection_directory}/*.fits') if "region" not in f.split('/')[-1]]

    # detected_coor = []
    # for detection in detected_files:
    #     filename = detection.split('/')[-1]
    #     source_type = filename.split('_')[0]
    #     source_coord = '_'.join(filename.split('_')[-2:]).replace('.fits', '')
    #     detected_coor.append([source_type, source_coord])

    # detected_coor = np.array(detected_coor)
    # detected_coor = np.unique(detected_coor, axis=0)

    # for coor in detected_coor:

    #     sources_coor = glob.glob(f'{postprocess_dir}{exo_dir}/{dynspec_folder}/detected_dynamic_spec/{coor[0]}_*_{coor[1]}.fits')
    #     sources_coor.sort()

    #     records = []

    #     for source in sources_coor:
    #         # Extract filename
    #         filename = source.split('/')[-1]
            
    #         # Extract time and frequency from filename
    #         time = float(filename.split('_')[2].replace('s', ''))
    #         freq = float(filename.split('_')[3].replace('kHz', ''))
            
    #         # Open FITS file to get SNR
    #         with fits.open(source) as hdu:
    #             transient_snr = hdu[0].header['SNR']
            
    #         # Append the time, freq, and SNR as a dictionary to the list
    #         records.append({
    #             'source': source,
    #             'time': time,
    #             'freq': freq,
    #             'snr': transient_snr
    #         })

    #     # Sort the list of dictionaries by SNR
    #     sorted_records = sorted(records, key=lambda x: x['snr'], reverse=True)

    #     # Get the record with the highest SNR
    #     highest_snr_record = sorted_records[0]

    #     # Extract the time and frequency corresponding to the highest SNR
    #     source_with_highest_snr = highest_snr_record['source']
    #     time_with_highest_snr = highest_snr_record['time']
    #     freq_with_highest_snr = highest_snr_record['freq']

    #     if freq_with_highest_snr > 195 * freq_windows[0]: # and time_with_highest_snr > 8 * time_windows[0]:

    #         with fits.open(source_with_highest_snr) as hdu:
    #             snr_map = hdu[0].data
    #             header = hdu[0].header

    #             # Time axis info
    #             crval1 = header['CRVAL1']
    #             cdelt1 = header['CDELT1']
    #             crpix1 = header['CRPIX1']
    #             naxis1 = header['NAXIS1']

    #             # Frequency axis info
    #             crval2 = header['CRVAL2']
    #             cdelt2 = header['CDELT2']
    #             crpix2 = header['CRPIX2']
    #             naxis2 = header['NAXIS2']

    #             # Calculate physical values for the axes
    #             time_vals = crval1 + (np.arange(naxis1) - (crpix1 - 1)) * cdelt1
    #             freq_vals = crval2 + (np.arange(naxis2) - (crpix2 - 1)) * cdelt2

    #             snr_map_no_nan = np.nan_to_num(snr_map, nan=0.0)

    #             filename = source_with_highest_snr.split('/')[-1]

    #             plt.figure(figsize=(12, 4))
    #             plt.imshow(snr_map_no_nan, aspect='auto', origin='lower', cmap='PiYG', vmin=-7, vmax=7, extent=[time_vals[0], time_vals[-1], freq_vals[0], freq_vals[-1]])
    #             cbar = plt.colorbar(shrink=0.95, aspect=15, pad=0.02)

    #             # Add a label to the colorbar and bring it closer
    #             cbar.set_label('SNR', rotation=270, labelpad=10)
    #             plt.xlabel('Time (s)')
    #             plt.ylabel('Frequency (MHz)')
    #             plt.title(f'SNR Map for {filename}')

    #             plt.savefig(f'{detection_directory}/{filename}.png', dpi=200, bbox_inches='tight')
    #             plt.close()

    # # Make a directory
    # cmd_png_dir = f'mkdir {postprocess_dir}{exo_dir}/{dynspec_folder}/v_png/'
    # subprocess.run(cmd_png_dir, shell=True, check=True)

    # png_files = glob.glob(f'{detection_directory}/*.png')
    # if png_files:
    #     # Only run the command if there are .png files
    #     cmd_mv_png = f'mv {detection_directory}/*.png {postprocess_dir}{exo_dir}/{dynspec_folder}/v_png/'
    #     subprocess.run(cmd_mv_png, shell=True, check=True)
    # else:
    #     print("No .png files found in the directory.")

    # # Move the png files to the directory
    # # cmd_mv_png = f'mv {detection_directory}/*.png {postprocess_dir}{exo_dir}/{dynspec_folder}/{exo_dir}_png/'
    # # subprocess.run(cmd_mv_png, shell=True, check=True)

    # # seventh, remove some directories within dynamic_spec
    # cmd_remo_dyna = f"rm -rf {postprocess_dir}/{exo_dir}/{dynspec_folder}/convol_gaussian {postprocess_dir}/{exo_dir}/{dynspec_folder}/noise_map" #{postprocess_dir}/{exo_dir}/{dynspec_folder}/weighted_dynamic_spec"
    # subprocess.run(cmd_remo_dyna, shell=True, check=True)

    # cmd_rename = f"mv {postprocess_dir}/{exo_dir}/{dynspec_folder}/detected_dynamic_spec {postprocess_dir}/{exo_dir}/{dynspec_folder}/detected_dynamic_spec_v"
    # subprocess.run(cmd_rename, shell=True, check=True)

    cmd_rename = f"mv {postprocess_dir}/{exo_dir}/{dynspec_folder}/weighted_dynamic_spec {postprocess_dir}/{exo_dir}/{dynspec_folder}/weighted_dynamic_spec_v"
    subprocess.run(cmd_rename, shell=True, check=True)

def source_find_i(exo_dir: str, time_windows, freq_windows):

    # get the folder name of the dynamic spectrum
    dynspec_folder = glob.glob(f'{postprocess_dir}{exo_dir}/dynamic_spec_*.MS')[0].split('/')[-1]

    # generate a MAD map to be used as a weight map in convolution
    # median_map, mad_map = generate_noise_map(f'{postprocess_dir}{exo_dir}/{dynspec_folder}/')
    mean_map, std_map = generate_noise_map_i(f'{postprocess_dir}{exo_dir}/{dynspec_folder}/')

    cmd_norm_dir = f'mkdir {postprocess_dir}{exo_dir}/{dynspec_folder}/weighted_dynamic_spec'
    subprocess.run(cmd_norm_dir, shell=True, check=True)
    generate_and_save_weight_map_i(f'{postprocess_dir}{exo_dir}/{dynspec_folder}/', f'{postprocess_dir}{exo_dir}/{dynspec_folder}/weighted_dynamic_spec/')

    # # mkdir to apply the Gaussian filter
    # cmd_convol_dir = f'mkdir {postprocess_dir}{exo_dir}/{dynspec_folder}/convol_gaussian/'
    # subprocess.run(cmd_convol_dir, shell=True, check=True)

    # # matched filtering
    # dynamic_directory = f'{postprocess_dir}{exo_dir}/{dynspec_folder}/weighted_dynamic_spec/'
    # convol_directory = f'{postprocess_dir}{exo_dir}/{dynspec_folder}/convol_gaussian/'

    # # get the size of the dynamic spectrum, to make sure that the windows do not exceed the size
    # dynspec_file = glob.glob(f'{dynamic_directory}/*.fits')[0]
    # with fits.open(dynspec_file) as hdul:
    #     dynspec_size = hdul[0].data.shape
    #     time_bins = dynspec_size[1]
    #     freq_bins = dynspec_size[0]

    # time_windows = [w for w in time_windows if w <= time_bins]
    # freq_windows = [w for w in freq_windows if w <= freq_bins]

    # convol_tasks = [delayed(apply_gaussian_filter)(filename, dynamic_directory, time_windows, freq_windows, convol_directory)
    #                    for filename in os.listdir(dynamic_directory)]
    
    # compute(*convol_tasks)

    # # generate noise map for the convolved dynamic spectrum
    # # but we need to make a directory for the noise map first
    # cmd_noise_dir = f'mkdir {postprocess_dir}{exo_dir}/{dynspec_folder}/noise_map/'
    # subprocess.run(cmd_noise_dir, shell=True, check=True)

    # noise_directory = f'{postprocess_dir}{exo_dir}/{dynspec_folder}/noise_map/'

    # cmd_detection_dir = f'mkdir {postprocess_dir}{exo_dir}/{dynspec_folder}/detected_dynamic_spec'
    # subprocess.run(cmd_detection_dir, shell=True, check=True)

    # detection_directory = f'{postprocess_dir}{exo_dir}/{dynspec_folder}/detected_dynamic_spec/'

    # # Not parallelized because it's opening too many files
    # for t_window in time_windows:
    #     for f_window in freq_windows:
    #         calculate_noise_for_window(convol_directory, noise_directory, t_window, f_window)

    # detection_tasks = [delayed(source_detection)(convol_directory, noise_directory, t_window, f_window, detection_directory, direction_threshold, direction_threshold_target, dynamic_threshold, dynamic_threshold_target)
    #         for t_window in time_windows
    #         for f_window in freq_windows]
    
    # compute(*detection_tasks)

    # detected_files = [f for f in glob.glob(f'{detection_directory}/*.fits') if "region" not in f.split('/')[-1]]

    # detected_coor = []
    # for detection in detected_files:
    #     filename = detection.split('/')[-1]
    #     source_type = filename.split('_')[0]
    #     source_coord = '_'.join(filename.split('_')[-2:]).replace('.fits', '')
    #     detected_coor.append([source_type, source_coord])

    # detected_coor = np.array(detected_coor)
    # detected_coor = np.unique(detected_coor, axis=0)

    # for coor in detected_coor:

    #     sources_coor = glob.glob(f'{postprocess_dir}{exo_dir}/{dynspec_folder}/detected_dynamic_spec/{coor[0]}_*_{coor[1]}.fits')
    #     sources_coor.sort()

    #     records = []

    #     for source in sources_coor:
    #         # Extract filename
    #         filename = source.split('/')[-1]
            
    #         # Extract time and frequency from filename
    #         time = float(filename.split('_')[2].replace('s', ''))
    #         freq = float(filename.split('_')[3].replace('kHz', ''))
            
    #         # Open FITS file to get SNR
    #         with fits.open(source) as hdu:
    #             transient_snr = hdu[0].header['SNR']
            
    #         # Append the time, freq, and SNR as a dictionary to the list
    #         records.append({
    #             'source': source,
    #             'time': time,
    #             'freq': freq,
    #             'snr': transient_snr
    #         })

    #     # Sort the list of dictionaries by SNR
    #     sorted_records = sorted(records, key=lambda x: x['snr'], reverse=True)

    #     # Get the record with the highest SNR
    #     highest_snr_record = sorted_records[0]

    #     # Extract the time and frequency corresponding to the highest SNR
    #     source_with_highest_snr = highest_snr_record['source']
    #     time_with_highest_snr = highest_snr_record['time']
    #     freq_with_highest_snr = highest_snr_record['freq']

    #     if freq_with_highest_snr > 195 * freq_windows[0]: # and time_with_highest_snr > 8 * time_windows[0]:

    #         with fits.open(source_with_highest_snr) as hdu:
    #             snr_map = hdu[0].data
    #             header = hdu[0].header

    #             # Time axis info
    #             crval1 = header['CRVAL1']
    #             cdelt1 = header['CDELT1']
    #             crpix1 = header['CRPIX1']
    #             naxis1 = header['NAXIS1']

    #             # Frequency axis info
    #             crval2 = header['CRVAL2']
    #             cdelt2 = header['CDELT2']
    #             crpix2 = header['CRPIX2']
    #             naxis2 = header['NAXIS2']

    #             # Calculate physical values for the axes
    #             time_vals = crval1 + (np.arange(naxis1) - (crpix1 - 1)) * cdelt1
    #             freq_vals = crval2 + (np.arange(naxis2) - (crpix2 - 1)) * cdelt2

    #             snr_map_no_nan = np.nan_to_num(snr_map, nan=0.0)

    #             filename = source_with_highest_snr.split('/')[-1]

    #             plt.figure(figsize=(12, 4))
    #             plt.imshow(snr_map_no_nan, aspect='auto', origin='lower', cmap='PiYG', vmin=-7, vmax=7, extent=[time_vals[0], time_vals[-1], freq_vals[0], freq_vals[-1]])
    #             cbar = plt.colorbar(shrink=0.95, aspect=15, pad=0.02)

    #             # Add a label to the colorbar and bring it closer
    #             cbar.set_label('SNR', rotation=270, labelpad=10)
    #             plt.xlabel('Time (s)')
    #             plt.ylabel('Frequency (MHz)')
    #             plt.title(f'SNR Map for {filename}')

    #             plt.savefig(f'{detection_directory}/{filename}.png', dpi=200, bbox_inches='tight')
    #             plt.close()

    # # Make a directory
    # cmd_png_dir = f'mkdir {postprocess_dir}{exo_dir}/{dynspec_folder}/i_png/'
    # subprocess.run(cmd_png_dir, shell=True, check=True)

    # png_files = glob.glob(f'{detection_directory}/*.png')
    # if png_files:
    #     # Only run the command if there are .png files
    #     cmd_mv_png = f'mv {detection_directory}/*.png {postprocess_dir}{exo_dir}/{dynspec_folder}/i_png/'
    #     subprocess.run(cmd_mv_png, shell=True, check=True)
    # else:
    #     print("No .png files found in the directory.")

    # # Move the png files to the directory
    # # cmd_mv_png = f'mv {detection_directory}/*.png {postprocess_dir}{exo_dir}/{dynspec_folder}/{exo_dir}_png/'
    # # subprocess.run(cmd_mv_png, shell=True, check=True)

    # # seventh, remove some directories within dynamic_spec
    # cmd_remo_dyna = f"rm -rf {postprocess_dir}/{exo_dir}/{dynspec_folder}/convol_gaussian {postprocess_dir}/{exo_dir}/{dynspec_folder}/noise_map" #{postprocess_dir}/{exo_dir}/{dynspec_folder}/weighted_dynamic_spec"
    # subprocess.run(cmd_remo_dyna, shell=True, check=True)

    # cmd_rename = f"mv {postprocess_dir}/{exo_dir}/{dynspec_folder}/detected_dynamic_spec {postprocess_dir}/{exo_dir}/{dynspec_folder}/detected_dynamic_spec_i"
    # subprocess.run(cmd_rename, shell=True, check=True)

    cmd_rename = f"mv {postprocess_dir}/{exo_dir}/{dynspec_folder}/weighted_dynamic_spec {postprocess_dir}/{exo_dir}/{dynspec_folder}/weighted_dynamic_spec_i"
    subprocess.run(cmd_rename, shell=True, check=True)


# Task 8. Clear up the directory

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

def exo_pipe(exo_dir, cal_dir, cal):
    # with open(lockfile, "w") as f:
    #     f.write("Processing ongoing")

    # task_copy_calibrator = copy_calibrator_data.submit(exo_dir)

    # task_copy_target = copy_target_data.submit(exo_dir)

    # Has calibrator been processed already? If not, find bad MA and do A-team calibration
    # cal, cal_dir, cali_check = task_copy_calibrator.result()

    if cali_check == False:
        bad_MAs = identify_bad_mini_arrays(cal, cal_dir)
        calibration_Ateam(cal, cal_dir, bad_MAs)
    else:
        with open(f'{postprocess_dir}/{cal_dir}/bad_MA.txt', 'r') as bad_MA_text:
            bad_MAs = bad_MA_text.read().strip()
        # No need to do Ateam calibration if it's done previously

    apply_Ateam_solution(cal_dir, exo_dir, bad_MAs)

    subtract_Ateam(exo_dir)

    dynspec(exo_dir)

    source_find_v(exo_dir, time_windows, freq_windows)

    source_find_i(exo_dir, time_windows, freq_windows)

    # clearup(exo_dir)

    # os.remove(lockfile)

# @flow(name='Check Flow', log_prints=True)
# def check_flow():
#     if os.path.exists(lockfile):
#         print("Exiting due to existence of lockfile")
#         return Completed(message="Lockfile exists, skipping run.")

#     new_data = check_new_data(watch_dir, postprocess_dir)

#     if len(new_data) > 0:
#         # Trigger the main flow
#         for unprocessed_data in new_data:
#             exo_pipe(unprocessed_data)

#         # exo_pipe(new_data[0])

#     return Completed(message="Run completed without issues.")

if __name__ == "__main__":
    exo_pipe(exo_dir, cal_dir, cal)