import subprocess
import os, glob
from astropy.io import fits
import numpy as np
from dask import delayed, compute
# from dask.distributed import Client, LocalCluster
from templates.Noise_esti_template import calculate_noise_for_window, apply_gaussian_filter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime, timedelta
from astropy.time import Time
from astropy.timeseries import LombScargle
from scipy.interpolate import griddata

###### Initial settings ######

postprocess_dir = "/data/xzhang/exo_img/"
period_dir = "/data/xzhang/exo_period/"
pipe_dir = "/home/xzhang/software/exo_img_pipe/"
target_name = "HD_189733"

# which bursting period to search
period_star = 11.94 # days
period_planet = 2.21857312 # days
period_min = period_planet/10.
period_max = period_planet*10.

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

def remove_excessive_nans(data, time):
    """Remove rows or columns with more than the threshold percentage of NaNs."""
    # Identify rows and columns with more than the threshold percentage of NaNs
    row_nan_fraction = np.mean(np.isnan(data), axis=1)  # Fraction of NaNs in each row
    col_nan_fraction = np.mean(np.isnan(data), axis=0)  # Fraction of NaNs in each column

    # Determine rows and columns to keep
    rows_to_keep = row_nan_fraction <= 0.7
    cols_to_keep = col_nan_fraction <= 0.3

    # Filter the data and time arrays
    data_clean = data[rows_to_keep, :]
    data_clean = data_clean[:, cols_to_keep]
    time_clean = time[cols_to_keep]  # Remove corresponding time points

    return data_clean, time_clean, np.where(~rows_to_keep)[0], np.where(~cols_to_keep)[0]

def interpolate_2d(data, x, y):
    """Interpolate NaNs in 2D using griddata."""
    # Create a grid of x and y indices
    grid_x, grid_y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

    # Flatten and filter valid points (non-NaN)
    valid_mask = ~np.isnan(data)
    points = np.array([grid_x[valid_mask], grid_y[valid_mask]]).T
    values = data[valid_mask]

    # Interpolate over the NaNs
    data_interpolated = griddata(points, values, (grid_x, grid_y), method='linear')

    # Fill any remaining NaNs (edges) with nearest neighbor interpolation
    data_interpolated[np.isnan(data_interpolated)] = griddata(points, values, (grid_x, grid_y), method='nearest')[np.isnan(data_interpolated)]

    return data_interpolated


# cluster = LocalCluster()
# client = Client(cluster)

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
period_search_dir = os.path.join(period_dir, target_name)
os.makedirs(period_search_dir, exist_ok=True)

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
                combined_time.append(time)

        combined_time = np.concatenate(combined_time) 
        combined_data = np.hstack(combined_data)  

        combined_data_clean, combined_time_clean, rows_removed, cols_removed = remove_excessive_nans(combined_data, combined_time)
        # Save the indices of removed rows and columns as complete lists
        with open(f'{period_dir}{target_name}/removed_rows_cols_{t_window_sec}s_{f_window_khz}kHz.txt', 'w') as f:
            f.write(f"Removed rows (frequency channels): {list(rows_removed)}\n")
            f.write(f"Removed columns (time points): {list(cols_removed)}\n")

        if combined_data_clean.size == 0:
            print(f"Skipping {t_window_sec}s {f_window_khz}kHz due to insufficient data after NaN removal.")
            continue

        combined_data_interpolated = interpolate_2d(combined_data_clean, np.arange(combined_data_clean.shape[1]), np.arange(combined_data_clean.shape[0]))

        ls = LombScargle(combined_time_clean, combined_data_interpolated[0, :])
        ls_freq, _ = ls.autopower(minimum_frequency=1/period_max, maximum_frequency=1/period_min)

        # set parameters for Lomb_Scargle periodogram
        lomb_scargle_matrix = np.zeros((combined_data_interpolated.shape[0], len(ls_freq)))
        # we need to get false alarm probability
        fap_matrix = np.zeros((combined_data_interpolated.shape[0], len(ls_freq)))

        @delayed
        def process_channel(freq_idx):
            """Function to process a single frequency channel with Lomb-Scargle."""
            y_data = combined_data_interpolated[freq_idx, :]  # Data for the current frequency channel
            ls = LombScargle(combined_time_clean, y_data)
            power = ls.power(ls_freq)
            # Calculate FAP for each power value
            fap_values = [ls.false_alarm_probability(p) for p in power]
            return power, fap_values

        # Parallel processing using Dask Delayed
        tasks = [process_channel(freq_idx) for freq_idx in range(combined_data_interpolated.shape[0])]
        results = compute(*tasks)

        # Unpack results into the matrices
        for i, (power, fap_values) in enumerate(results):
            lomb_scargle_matrix[i, :] = power
            fap_matrix[i, :] = fap_values

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

        # plot an image for the periodogram
        all_frequencies = np.linspace(freq_min, freq_min + (num_chan - 1) * delta_freq, num_chan)
        frequencies = np.delete(all_frequencies, rows_removed)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        # Plot Lomb-Scargle power (top panel)
        c1 = ax1.imshow(lomb_scargle_matrix, aspect='auto', origin='lower', 
                        extent=[ls_freq[0], ls_freq[-1], frequencies[0], frequencies[-1]],
                        cmap='viridis')
        fig.colorbar(c1, ax=ax1, label='Lomb-Scargle Power')
        ax1.set_ylabel('Radio Frequency (MHz)')
        ax1.set_title(f'Power (t_window = {t_window_sec}s, f_window = {f_window_khz}kHz)')

        # Plot vertical lines for star and planet periods
        star_period_freq = 1 / period_star
        planet_period_freq = 1 / period_planet
        ax1.axvline(star_period_freq, color='tab:red', linestyle='--', label='Star Period')
        ax1.axvline(planet_period_freq, color='tab:blue', linestyle='--', label='Planet Period')

        # Add labels for the star and planet periods
        ax1.text(star_period_freq, frequencies[-1], 'Star Period', color='red', fontsize=10, ha='right', va='top', rotation=90)
        ax1.text(planet_period_freq, frequencies[-1], 'Planet Period', color='blue', fontsize=10, ha='right', va='top', rotation=90)
        
        ax1.legend()

        # Plot FAP (bottom panel)
        c2 = ax2.imshow(fap_matrix, aspect='auto', origin='lower',
                        extent=[ls_freq[0], ls_freq[-1], frequencies[0], frequencies[-1]],
                        cmap='plasma', vmin=0, vmax=1)
        fig.colorbar(c2, ax=ax2, label='False Alarm Probability')
        ax2.set_xlabel('Lomb-Scargle Frequency (cycles/day)')
        ax2.set_ylabel('Radio Frequency (MHz)')
        ax2.set_title('False Alarm Probability')

        # Plot vertical lines for star and planet periods
        ax2.axvline(star_period_freq, color='red', linestyle='--')
        ax2.axvline(planet_period_freq, color='blue', linestyle='--')

        # Add labels for the star and planet periods on FAP plot
        ax2.text(star_period_freq, frequencies[-1], 'Star Period', color='red', fontsize=10, ha='right', va='top', rotation=90)
        ax2.text(planet_period_freq, frequencies[-1], 'Planet Period', color='blue', fontsize=10, ha='right', va='top', rotation=90)

        # Save the plot
        plt.tight_layout()
        plt.savefig(f'{period_dir}{target_name}/lomb_scargle_plot_{t_window_sec}s_{f_window_khz}kHz.png', dpi=300, bbox_inches='tight', facecolor='w')
        plt.close()
        print(f"Saved plot for {t_window_sec}s_{f_window_khz}kHz")




