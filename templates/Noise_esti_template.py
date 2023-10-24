# Here's the code rewritten as a function. Note that you would need to import the necessary libraries in your local environment.

from astropy.io import fits  # Import within the function; you'll need to import this in your local environment
import numpy as np
import os, glob
# from scipy.ndimage import gaussian_filter
# from astropy.convolution import Gaussian2DKernel, convolve
from scipy.ndimage import gaussian_filter

def generate_noise_map(dynspec_directory):

    subsample_spectra = []

    fits_directory = f'{dynspec_directory}/TARGET/'

    for filename in os.listdir(fits_directory):
        if filename.endswith('.fits'):
            filepath = os.path.join(fits_directory, filename)

            # Open the FITS file
            with fits.open(filepath) as hdul:
                # Check if the file is part of the subsample based on the 'SRC-TYPE' header parameter
                if hdul[0].header.get('SRC-TYPE', '').strip() == 'Field':
                    # Assuming the dynamic spectrum for Stokes I is in the first HDU
                    # This may vary depending on how your data is structured
                    data = hdul[0].data[3, :, :]

                    # Add this dynamic spectrum to our list
                    subsample_spectra.append(data)

    median_map = np.median(subsample_spectra, axis=0)
    mad_map = np.median(np.abs(subsample_spectra - median_map), axis=0)

    with fits.open(filepath) as hdul:
        hdul[0].data[0,:,:] = median_map
        hdul[0].data[1,:,:] = mad_map
        hdul[0].data[2,:,:] = 0
        hdul[0].data[3,:,:] = 0

        hdul.writeto(f'{dynspec_directory}/median_mad.fits', overwrite=True)

    return median_map, mad_map

def generate_and_save_snr_map(dynspec_directory, snr_fits_directory):
    """
    Generate and save SNR maps for each FITS file in the directory.
    Parameters:
    - fits_directory: str
        The directory containing the original FITS files.
    - median_fits_path: str
        Path to the median map FITS file.
    - mad_fits_path: str
        Path to the MAD map FITS file.
    - snr_fits_directory: str
        Directory where the SNR maps will be saved.
    """

    # Read the median and MAD maps from the FITS files
    with fits.open(f'{dynspec_directory}/median_mad.fits') as hdul:
        median_map = hdul[0].data[0,:,:]
        mad_map = hdul[0].data[1,:,:]
    fits_directory = f'{dynspec_directory}/TARGET/'
    # Loop through each FITS file in the directory
    for filename in os.listdir(fits_directory):
        if filename.endswith('.fits'):
            filepath = os.path.join(fits_directory, filename)
            
            # Open the FITS file
            with fits.open(filepath) as hdul:
                # Extract Stokes V data (assuming it's the 4th index in the first dimension)
                stokes_v_data = hdul[0].data[3, :, :]
                
                # Calculate the SNR map
                snr_map = (stokes_v_data - median_map) / mad_map
                
                # Prepare the HDU for the SNR map
                snr_hdu = fits.PrimaryHDU(snr_map)
                snr_hdu.header = hdul[0].header.copy()
                # Remove the polarization axis information
                for key in ['NAXIS3', 'CTYPE3', 'CRVAL3', 'CDELT3', 'CRPIX3', 'CUNIT3']:
                    snr_hdu.header.remove(key, ignore_missing=True)
                snr_hdu.header['NAXIS'] = 2  # Now it's a 2D image
                
                # Save the SNR map as a FITS file
                snr_fits_path = os.path.join(snr_fits_directory, f"SNR_{filename}")
                snr_hdu.writeto(snr_fits_path, overwrite=True)

def apply_gaussian_filter(filename, dynamic_directory, time_windows, freq_windows, convol_directory):
   
    # transient_detected_files = []
    
    # Loop through each SNR FITS file in the directory
    # for filename in os.listdir(snr_fits_directory):
        # if filename.startswith('SNR_') and filename.endswith('.fits'):
    filepath = os.path.join(dynamic_directory, filename)
    
    # Open the SNR FITS file
    with fits.open(filepath) as hdul:
        # Check if this is a target
        # is_target = hdul[0].header.get('SRC-TYPE', '').strip() == 'Target'
        
        # Extract dynamic spec
        dynspec_data = hdul[0].data
        
        # Remove NaN values (replace with zeros)
        dynspec_data = np.nan_to_num(dynspec_data)
        
        # # Get the dimensions of the dynamic spectrum
        # time_bins, freq_bins = dynspec_data.shape
        
        # # Filter out window sizes that are larger than the dynamic spectrum
        # time_windows = [w for w in time_windows if w <= time_bins]
        # freq_windows = [w for w in freq_windows if w <= freq_bins]
        
        # Loop through each combination of time and frequency window
        for t_window in time_windows:
            for f_window in freq_windows:
                # # Apply Gaussian filter
                sigma_t = t_window / 6  # Standard deviation for time
                sigma_f = f_window / 6  # Standard deviation for frequency
                # filtered_snr = gaussian_filter(snr_data, sigma=[sigma_f, sigma_t])

                sigma = [sigma_f, sigma_t]

                convol_data = gaussian_filter(dynspec_data, sigma=sigma)

                kernel_t = int(sigma_t * 3)
                kernel_f = int(sigma_f * 3)

                convol_data[:kernel_f, :] = np.nan
                convol_data[-kernel_f:, :] = np.nan
                convol_data[:, :kernel_t] = np.nan
                convol_data[:, -kernel_t:] = np.nan

                # gaussian_kernel = Gaussian2DKernel(sigma_t, sigma_f)

                # convol_data = convolve(dynspec_data, gaussian_kernel, boundary='extend')

                # scale_factor = (f_window * t_window) ** 0.5

                # filtered_data *= scale_factor

                # We need to normalize the filtered SNR map to account for the different window sizes
                # normal_filtered_snr = filtered_snr * ( 2 * np.pi * sigma_t * sigma_f) ** 0.5 
                
                # Flag potential transients
                # filtered_snr_threshold = snr_threshold / ( (2 * np.pi * sigma_t ** 2) ** 0.5 * (2 * np.pi * sigma_f ** 2) ** 0.5 )
                # transient_detected = np.any(filtered_data >= snr_threshold)
                # if transient_detected:
                #     transient_detected_files.append(filename)

                t_window_sec = t_window * 8
                f_window_khz = f_window * 60
                
                # Save the filtered SNR map based on conditions
                prefix = "convol"
                # Prepare the HDU for the filtered SNR map
                convol_hdu = fits.PrimaryHDU(convol_data)
                convol_hdu.header = hdul[0].header.copy()
                
                # Save the filtered SNR map as a FITS file
                output_filename = f"{prefix}_{t_window_sec}s_{f_window_khz}kHz_{filename}"
                output_filepath = os.path.join(convol_directory, output_filename)
                convol_hdu.writeto(output_filepath, overwrite=True)
                # convol_hdu.close()
                            
    # return transient_detected_files

# Usage example (you'll run this part in your local environment)
# Time windows in units of 8 sec bins: [4, 8, 16, ..., 1024] => [32, 64, 128, ..., 8192] sec
# Frequency windows in units of 60 kHz bins: [8, 16, ..., 512] => [480, 960, ..., 30720] kHz
# snr_threshold = 5  # An example value; you may need to adjust this based on your specific needs
# detected_files = matched_filtering_with_detection_v3('/path/to/snr/fits', list(range(4, 1025, 4)), list(range(8, 513, 8)), '/path/to/save/filtered/fits', snr_threshold)
# print("Files where potential transients were detected:", detected_files)

def calculate_noise_for_window(convol_directory, noise_directory, t_window, f_window):
    """
    Generate a time-frequency noise map based on a subsample of FITS files.

    Parameters:
    - fits_directory: str
        The directory containing the FITS files.

    Returns:
    - median_map: np.ndarray
        2D array representing the median noise level at each time-frequency pixel.
    - mad_map: np.ndarray
        2D array representing the Median Absolute Deviation at each time-frequency pixel.
    """

    # Initialize lists to store the dynamic spectra from the subsample of directions
    subsample_spectra = []

    t_window_sec = t_window * 8

    # for f_window in freq_windows:
    f_window_khz = f_window * 60

    # Loop through each FITS file in the directory
    for filepath in glob.glob(f'{convol_directory}/convol_{t_window_sec}s_{f_window_khz}kHz*.fits'):
        with fits.open(filepath) as hdul:
            # Check if the file is part of the subsample based on the 'SRC-TYPE' header parameter
            if hdul[0].header.get('SRC-TYPE', '').strip() == 'Field':
                # Assuming the dynamic spectrum for Stokes I is in the first HDU
                # This may vary depending on how your data is structured
                data = hdul[0].data
                
                # Add this dynamic spectrum to our list
                subsample_spectra.append(data)

    # Convert the list of 2D arrays into a 3D NumPy array
    # The shape would be (num_directions, num_time_bins, num_freq_channels)
    subsample_spectra = np.array(subsample_spectra)

    # Calculate the median and MAD along the direction axis (axis=0)
    median_map = np.median(subsample_spectra, axis=0)
    mad_map = np.median(np.abs(subsample_spectra - median_map), axis=0)

    with fits.open(filepath) as hdul:
        hdul[0].data = median_map
        hdul.writeto(f'{noise_directory}/median_{t_window_sec}s_{f_window_khz}kHz.fits', overwrite=True)

    with fits.open(filepath) as hdul:
        hdul[0].data = mad_map
        hdul.writeto(f'{noise_directory}/mad_{t_window_sec}s_{f_window_khz}kHz.fits', overwrite=True)

def source_detection(convol_directory, noise_directory, t_window, f_window, detection_directory, snr_threshold):

    t_window_sec = t_window * 8
    f_window_khz = f_window * 60

    with fits.open(f'{noise_directory}/median_{t_window_sec}s_{f_window_khz}kHz.fits') as hdul:
        median_map = hdul[0].data

    with fits.open(f'{noise_directory}/mad_{t_window_sec}s_{f_window_khz}kHz.fits') as hdul:
        mad_map = hdul[0].data

    for filepath in glob.glob(f'{convol_directory}/convol_{t_window_sec}s_{f_window_khz}kHz*.fits'):
        filename = filepath.split('/')[-1]
        with fits.open(filepath) as hdul:
            convol_data = hdul[0].data
            snr_map = (convol_data - median_map) / mad_map
            source_type = hdul[0].header.get('SRC-TYPE', '').strip()
            # is_target = hdul[0].header.get('SRC-TYPE', '').strip() == 'Target'
            source_detected = np.any(np.abs(snr_map) >= snr_threshold)
            source_region = snr_map[np.abs(snr_map) >= snr_threshold]

            if source_detected:
                snr_median = np.nanmedian(snr_map)
                snr_mad = np.nanmedian(np.abs(snr_map - snr_median))
                transient_detected = np.any(np.abs(source_region - snr_median)/snr_mad >= snr_threshold)
                prefix = "transient" if transient_detected else "source"
                snr_hdu = fits.PrimaryHDU(snr_map)
                snr_hdu.header = hdul[0].header.copy()

                output_filename = f"{prefix}_{source_type}_{filename}"
                output_filepath = os.path.join(detection_directory, output_filename)
                snr_hdu.writeto(output_filepath, overwrite=True)

                source_region_hdu = fits.PrimaryHDU(source_region)
                source_region_hdu.header = hdul[0].header.copy()
                region_output_filename = f"{prefix}_{source_type}_region_{filename}"
                region_output_filepath = os.path.join(detection_directory, region_output_filename)
                source_region_hdu.writeto(region_output_filepath, overwrite=True)