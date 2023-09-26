import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.simbad import Simbad

def simbad_coor(star_name):
    # Create a Simbad object
    simbad_query = Simbad()

    # Query Simbad for the star
    result = simbad_query.query_object(star_name)

    # Extract ra and dec from the result in the string format
    ra_str = result['RA'][0]
    dec_str = result['DEC'][0]

    # Convert the ra and dec to degrees
    coord = SkyCoord(ra_str + " " + dec_str, unit=(u.hourangle, u.deg))

    ra_deg = coord.ra.degree
    dec_deg = coord.dec.degree

    return ra_deg, dec_deg

def make_target_list(target_name, postprocess_dir, exo_dir):
    

    # Define the center of the field of view
    ra_center, dec_center = simbad_coor(target_name)  # in degrees

    # Define the radius of the field of view
    fov_radius = 20.0 / 2.0  # in degrees

    # Define the maximum angular distance between points
    max_angular_distance = 0.5  # in degrees

    # Define the radius of the sphere (the celestial sphere)
    # Since we are interested in angular distances, we can set R = 1
    R = 1

    # Convert the max_angular_distance to radians
    max_angular_distance_radians = np.radians(max_angular_distance)

    # Calculate the total surface area of the sphere
    sphere_area = 4 * np.pi * R**2

    # Calculate the area per point based on the max_angular_distance
    area_per_point = np.pi * (max_angular_distance_radians / 2)**2

    # Estimate the number of points needed to cover the whole sphere
    num_points = int(np.round(sphere_area / area_per_point))

    # Generate points using Fibonacci Sphere algorithm
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - indices / num_points * 2)  # in radians
    theta = np.pi * (1 + 5**0.5) * indices  # in radians

    # Convert spherical coordinates to cartesian coordinates
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)

    # Convert cartesian coordinates to right ascension and declination
    ra = np.arctan2(y, x)  # in radians
    dec = np.arcsin(z)  # in radians

    # Convert right ascension and declination from radians to degrees
    ra = np.degrees(ra)
    dec = np.degrees(dec)

    # Filter points within the field of view
    coords = SkyCoord(ra * u.degree, dec * u.degree, frame='icrs')
    center = SkyCoord(ra_center * u.degree, dec_center * u.degree, frame='icrs')
    within_fov = center.separation(coords).degree < fov_radius
    ra_within_fov = ra[within_fov]
    dec_within_fov = dec[within_fov]

    # Plot the points within the field of view
    # plt.figure(figsize=(10, 6))
    # plt.scatter(ra_within_fov, dec_within_fov, s=10)
    # plt.xlabel('Right Ascension (deg)')
    # plt.ylabel('Declination (deg)')
    # plt.title('Points within Field of View')
    # plt.grid(True)
    # plt.show()

    # Open a file in write mode
    with open(f'{postprocess_dir}/{exo_dir}/target.txt', 'w') as file:
        # Targets first
        file.write(f"{target_name}, {ra_center:.6f}, {dec_center:.6f}, Target\n")

        if target_name == "KEPLER_42":
            ra_1, dec_1 = simbad_coor('KEPLER_78')
            ra_2, dec_2 = simbad_coor('KOI-55')
            ra_3, dec_3 = simbad_coor('KOI-4777')
            ra_4, dec_4 = simbad_coor('KEPLER_32')

        file.write(f"KEPLER_78, {ra_1:.6f}, {dec_1:.6f}, Target\n")
        file.write(f"KOI-55, {ra_2:.6f}, {dec_2:.6f}, Target\n")
        file.write(f"KOI-4777, {ra_3:.6f}, {dec_3:.6f}, Target\n")
        file.write(f"KEPLER_32, {ra_4:.6f}, {dec_4:.6f}, Target\n")

        # Loop through each point within the field of view
        for i, (ra, dec) in enumerate(zip(ra_within_fov, dec_within_fov)):
            # Wrap RA within [0, 360] degrees
            ra = ra % 360
            # Write the point index, RA, and Dec to the file in the specified format
            file.write(f"Point_{i}, {ra:.6f}, {dec:.6f}, Field\n")

    print(f"{len(ra_within_fov)} points have been written to target.txt")


