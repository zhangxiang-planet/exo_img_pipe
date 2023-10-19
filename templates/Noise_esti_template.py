# Here's the code rewritten as a function. Note that you would need to import the necessary libraries in your local environment.

def generate_noise_map(dynspec_directory):
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
    from astropy.io import fits  # Import within the function; you'll need to import this in your local environment
    import numpy as np
    import os

    # Initialize lists to store the dynamic spectra from the subsample of directions
    subsample_spectra = []

    fits_directory = f'{dynspec_directory}/TARGET/'

    # Loop through each FITS file in the directory
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

    # Convert the list of 2D arrays into a 3D NumPy array
    # The shape would be (num_directions, num_time_bins, num_freq_channels)
    subsample_spectra = np.array(subsample_spectra)

    # Calculate the median and MAD along the direction axis (axis=0)
    median_map = np.median(subsample_spectra, axis=0)
    mad_map = np.median(np.abs(subsample_spectra - median_map), axis=0)

    hdul[0].data[0,:,:] = median_map
    hdul[0].data[1,:,:] = mad_map
    hdul[0].data[2,:,:] = 0
    hdul[0].data[3,:,:] = 0

    hdul.writeto(f'{dynspec_directory}/median_mad.fits', overwrite=True)

    return median_map, mad_map

# Usage example (you'll run this part in your local environment)
# median_map, mad_map = generate_noise_map('/path/to/your/directory')
