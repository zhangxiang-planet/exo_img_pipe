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
watch_dir = "/databf/nenufar-nri/LT02/2025/??/*"

preprocess_dir = "/databf/nenufar-nri/LT02/"
postprocess_dir = "/data/xzhang/exo_img/"
pipe_dir = "/home/xzhang/software/exo_img_pipe/"
lockfile = "/home/xzhang/software/exo_img_pipe/lock.file"
singularity_file = "/home/xzhang/software/ddf_dev2_ateam.sif"
skip_file = "/home/xzhang/software/exo_img_pipe/templates/skip.txt"

# Calibrators
CALIBRATORS = ['CYG_A', 'CAS_A', 'TAU_A', 'VIR_A']

# How many SB per processing chunk
# chunk_num = 12

# How many channels per SB
chan_per_SB_origin = 2
ave_chan = 1
chan_per_SB = int(chan_per_SB_origin/ave_chan)
ave_time = 8

# chan_per_SB = 12

# Avoid bad channel making KMS hang
# bin_per_MSB = chunk_num // 3

# the lowest SB we use
SB_min = 106 # 92
SB_max = 401
SB_ave_kms = 2

# The region file we use for A-team removal
region_file = "/home/xzhang/software/exo_img_pipe/regions/Ateam.reg"

# Window and SNR threshold for matched filtering
direction_threshold = 6
direction_threshold_target = 5
dynamic_threshold = 6
dynamic_threshold_target = 5
# snr_threshold = 7
# snr_threshold_target = 6
time_windows = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
freq_windows = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
# 30 sec (5 min) 
# 600 kHz (6 MHz)

###### Lock the flow runs when data processing is ongoing ######

# class LockExitException(Exception):
#     pass

###### Here are the tasks (aka functions doing the job) ######

# Task 0. Find un-processed data directories

exo_dir = "20231104_184200_20231104_220000_KELT-1_TRACKING"


def source_find_v(exo_dir: str, time_windows, freq_windows, origin: bool = False):

    if origin:
        dynspec_folder = glob.glob(f'{postprocess_dir}{exo_dir}/dynamic_spec_origins_*.MS')[0].split('/')[-1]
    else:
        # get the folder name of the dynamic spectrum
        dynspec_folder = glob.glob(f'{postprocess_dir}{exo_dir}/dynamic_spec_Dyn*.MS')[0].split('/')[-1]

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

        # for record in sorted_records[1:]:
        #     os.remove(record['source'])

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