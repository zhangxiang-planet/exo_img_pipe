import subprocess
import os, glob
from astropy.io import fits
import numpy as np
from dask import delayed, compute
from templates.Noise_esti_template import calculate_noise_for_window, apply_gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime, timedelta
from astropy.time import Time
from astropy.timeseries import LombScargle

###### Initial settings ######

postprocess_dir = "/data/xzhang/exo_img/"
period_dir = "/data/xzhang/exo_period/"
pipe_dir = "/home/xzhang/software/exo_img_pipe/"
target_name = "HD_189733"

freq_min = 21.09222412109375 # MHz
delta_freq = 0.1953125 # MHz
num_chan = 212

delta_time = 8.053063089028 # seconds

# snr_threshold = 7
# snr_threshold_target = 6
time_windows = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
freq_windows = [0.25, 0.5, 1, 2, 4, 8, 16, 32]

# list target observations
target_dirs = glob.glob(postprocess_dir + "*" + target_name + "_TRACKING")
target_dirs.sort()

# # convolve the dynamic spectra within each observation
# for target_dir in target_dirs:
#     print("Processing " + target_dir)
#     dynspec_folder = target_dir + "/dynamic_spec_DynSpecs_GSB.MS/"

#     # mkdir to apply the Gaussian filter
#     cmd_convol_dir = f'mkdir {dynspec_folder}/convol_gaussian/'
#     subprocess.run(cmd_convol_dir, shell=True, check=True)

#     dynamic_directory = f'{dynspec_folder}/weighted_dynamic_spec_v/'
#     convol_directory = f'{dynspec_folder}/convol_gaussian/'

#     # get the size of the dynamic spectrum, to make sure that the windows do not exceed the size
#     dynspec_file = glob.glob(f'{dynamic_directory}/*.fits')[0]
#     with fits.open(dynspec_file) as hdul:
#         dynspec_size = hdul[0].data.shape
#         time_bins = dynspec_size[1]
#         freq_bins = dynspec_size[0]

#     time_windows = [w for w in time_windows if w <= time_bins]
#     freq_windows = [w for w in freq_windows if w <= freq_bins]

#     convol_tasks = [delayed(apply_gaussian_filter)(filename, dynamic_directory, time_windows, freq_windows, convol_directory)
#                        for filename in os.listdir(dynamic_directory)]
    
#     compute(*convol_tasks)

#     cmd_noise_dir = f'mkdir {dynspec_folder}/noise_map/'
#     subprocess.run(cmd_noise_dir, shell=True, check=True)

#     noise_directory = f'{dynspec_folder}/noise_map/'

#     # we modify this part to get convolved dynamic spectra

#     cmd_detection_dir = f'mkdir {dynspec_folder}/convolved_dynamic_spec'
#     subprocess.run(cmd_detection_dir, shell=True, check=True)

#     detection_directory = f'{dynspec_folder}/convolved_dynamic_spec/'

#     # Not parallelized because it's opening too many files
#     for t_window in time_windows:
#         for f_window in freq_windows:
#             calculate_noise_for_window(convol_directory, noise_directory, t_window, f_window)

#             t_window_sec = t_window * 8
#             f_window_khz = f_window * 195

#             with fits.open(f'{noise_directory}/mean_{t_window_sec}s_{f_window_khz}kHz.fits') as hdul:
#                 mean_map = hdul[0].data

#             with fits.open(f'{noise_directory}/std_{t_window_sec}s_{f_window_khz}kHz.fits') as hdul:
#                 std_map = hdul[0].data

#             for filepath in glob.glob(f'{convol_directory}/convol_{t_window_sec}s_{f_window_khz}kHz*.fits'):
#                 filename = filepath.split('/')[-1]
#                 with fits.open(filepath) as hdul:
#                     convol_data = hdul[0].data
#                     # snr_map = (convol_data - median_map) / mad_map
#                     # replace with mean and std
#                     snr_map = (convol_data - mean_map) / std_map
#                     source_type = hdul[0].header.get('SRC-TYPE', '').strip()
#                     is_target = hdul[0].header.get('SRC-TYPE', '').strip() == 'Target'
#                     if is_target:
#                         snr_hdu = fits.PrimaryHDU(snr_map)
#                         snr_hdu.header = hdul[0].header.copy()

#                         output_filename = f"{source_type}_{filename}"
#                         output_filepath = os.path.join(detection_directory, output_filename)
#                         snr_hdu.writeto(output_filepath, overwrite=True)

#     cmd_remo_dyna = f"rm -rf {dynspec_folder}/convol_gaussian {dynspec_folder}/noise_map" #{postprocess_dir}/{exo_dir}/{dynspec_folder}/weighted_dynamic_spec"
#     subprocess.run(cmd_remo_dyna, shell=True, check=True)

# make the period search directory
cmd_period_dir = f'mkdir {period_dir}{target_name}'
subprocess.run(cmd_period_dir, shell=True, check=True)

for t_window in time_windows:
    for f_window in freq_windows:
        t_window_sec = t_window * 8
        f_window_khz = f_window * 195
        detection_files = glob.glob(f'{postprocess_dir}*{target_name}_TRACKING/dynamic_spec_DynSpecs_GSB.MS/convolved_dynamic_spec/*_{t_window_sec}s_{f_window_khz}kHz*.fits')
        detection_files.sort()
        print(f"Processing {t_window_sec}s {f_window_khz}kHz")

        combined_time = []
        combined_data = []

        for detection_file in detection_files:
            with fits.open(detection_file) as hdul:
                file_chan = hdul[0].header['NAXIS2']
                data = hdul[0].data
                resampled_data = np.full((num_chan, data.shape[1]), np.nan)
                resampled_data[0:file_chan, :] = data
                combined_data.append(resampled_data)

                obs_start_str = hdul[0].header['OBS-STAR']
                obs_start_time = datetime.strptime(obs_start_str, '%Y-%m-%dT%H:%M:%S.%f')
                ref_time = Time(obs_start_time).mjd
                time_offset = np.arange(hdul[0].header['NAXIS1']) * hdul[0].header['CDELT1']
                time = ref_time + time_offset / 86400.0
                time_array = np.tile(time, (num_chan, 1))
                combined_time.append(time_array)

        combined_time = np.hstack(combined_time) 
        combined_data = np.hstack(combined_data)  

        ls = LombScargle(combined_time[0, :], combined_data[0, :])
        ls_freq, _ = ls.autopower()

        # set parameters for Lomb_Scargle periodogram
        lomb_scargle_matrix = np.zeros((num_chan, len(ls_freq)))
        # we need to get false alarm probability
        fap_matrix = np.zeros((num_chan, len(ls_freq)))

        # for i in range(num_chan):
        #     freq, power = LombScargle(combined_time, combined_data[i]).autopower()
        #     lomb_scargle_matrix[i] = power
        #     fap = LombScargle(combined_time, combined_data[i]).false_alarm_probability(power.max())
        #     fap_matrix[i] = fap

        for freq_idx in range(num_chan):
            ls = LombScargle(combined_time, combined_data[freq_idx, :])
            power = ls.power(ls_freq)
            lomb_scargle_matrix[freq_idx, :] = power
            
            # Calculate the FAP for the entire power spectrum
            for i, p in enumerate(power):
                fap_matrix[freq_idx, i] = ls.false_alarm_probability(p)

        # save the Lomb-Scargle periodogram into fits files
        header = fits.Header()
        header['COMMENT'] = "Lomb-Scargle power matrix"
        hdu_power = fits.PrimaryHDU(data=lomb_scargle_matrix, header=header)
        hdu_power.writeto(f'{period_dir}{target_name}/power_{t_window_sec}s_{f_window_khz}kHz.fits', overwrite=True)

        header = fits.Header()
        header['COMMENT'] = "False alarm probability matrix"
        hdu_fap = fits.PrimaryHDU(data=fap_matrix, header=header)
        hdu_fap.writeto(f'{period_dir}{target_name}/fap_{t_window_sec}s_{f_window_khz}kHz.fits', overwrite=True)

        # save the frequency list in a text file
        np.savetxt(f'{period_dir}{target_name}/freq_{t_window_sec}s_{f_window_khz}kHz.txt', ls_freq)


