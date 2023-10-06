import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astropy.io import fits

def plot_target_distribution(postprocess_dir, exo_dir):

    target_list = np.genfromtxt(f'{postprocess_dir}/{exo_dir}/target.txt', dtype='str', delimiter=',')
    sub_img = f'{postprocess_dir}/{exo_dir}/Image_SUB.app.restored.fits'

    data = fits.getdata(sub_img)
    wcs = WCS(fits.getheader(sub_img)).dropaxis(3).dropaxis(2)

    target = target_list[0]
    exo_list = target_list[target_list[:, 3] == ' Exoplanet']
    ucd_list = target_list[target_list[:, 3] == ' UCD']
    field_list = target_list[target_list[:, 3] == ' Field']

    coor_target = SkyCoord(target[1].astype(float), target[2].astype(float), unit=(u.degree, u.degree))
    coor_exo = SkyCoord(exo_list[:, 1].astype(float), exo_list[:, 2].astype(float), unit=(u.degree, u.degree))
    coor_ucd = SkyCoord(ucd_list[:, 1].astype(float), ucd_list[:, 2].astype(float), unit=(u.degree, u.degree))
    coor_field = SkyCoord(field_list[:, 1].astype(float), field_list[:, 2].astype(float), unit=(u.degree, u.degree))

    x_target, y_target = wcs.world_to_pixel(coor_target)
    x_exo, y_exo = wcs.world_to_pixel(coor_exo)
    x_ucd, y_ucd = wcs.world_to_pixel(coor_ucd)
    x_field, y_field = wcs.world_to_pixel(coor_field)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=wcs)

    img = ax.imshow(data[0,0], cmap='Blues', vmin=-15, vmax=40, zorder=0)

    ax.grid(color='white', ls='--', zorder=1)

    ax.scatter(x_field, y_field, s=3, marker='.', color='gray', label='Field', zorder=2)
    ax.scatter(x_ucd, y_ucd, s=5, marker='*', color='yellow', label='UCD', zorder=2)
    ax.scatter(x_exo, y_exo, s=5, marker='+', color='tab:orange', label='Exoplanet', zorder=2)
    ax.scatter(x_target, y_target, s=5, marker='o', color='tab:red', label='Target', zorder=2)


    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')

    ax.legend(loc='upper right')

    cbar = plt.colorbar(img, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label('Intensity (Jy)')  # You can set this label to something more descriptive if needed.

    img_file = f'{postprocess_dir}/{exo_dir}/target_distri.png'

    plt.savefig(img_file, dpi=300, bbox_inches='tight')
