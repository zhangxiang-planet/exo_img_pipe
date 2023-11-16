import subprocess
import os, glob
from astropy.io import fits
import numpy as np

preprocess_dir = "/databf/nenufar-nri/LT02/"
postprocess_dir = "/data/xzhang/exo_img/"
pipe_dir = "/home/xzhang/software/exo_img_pipe/"
singularity_file = "/home/xzhang/software/ddf.sif"

# Calibrators
CALIBRATORS = ['CYG_A', 'CAS_A', 'TAU_A', 'VIR_A']

# How many SB per processing chunk
chunk_num = 12

ave_chan = 4

chan_per_SB = 12

bin_per_SB = chan_per_SB // ave_chan

# the lowest SB we use
SB_min = 92

######################
# A region grow function here
def grow_region(image, mask, start_coords):
    region = set()
    to_check = set(start_coords)

    # Initialize min/max coordinates for the bounding box
    min_x = max_x = start_coords[0][1]
    min_y = max_y = start_coords[0][0]

    while to_check:
        y, x = to_check.pop()
        if mask[y, x] and (y, x) not in region:
            region.add((y, x))

            # Update min/max x and y
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)

            # Add neighboring pixels to check
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy != 0 or dx != 0:
                        new_y, new_x = y + dy, x + dx
                        if 0 <= new_y < image.shape[0] and 0 <= new_x < image.shape[1]:
                            to_check.add((new_y, new_x))
    return region, (min_y, max_y, min_x, max_x)


######################
# Watch the directory of selected images
watch_dir = "/data/xzhang/exo_candidates/"

# get a list of images to be inspected
img_list = glob.glob(watch_dir + "*_png/*.png")
img_list.sort()

# loop over the images
for img in img_list:
    # do we have a corresponding fits file?
    image_file = img.replace(".fits.png", ".image.fits")
    if not os.path.exists(image_file):
        exo_dir = img.split("/")[-2].replace("_png", "")
        img_name = img.split("/")[-1].replace(".png", "")
        dyna_file = glob.glob(f'{postprocess_dir}{exo_dir}/dynamic_spec_DynSpecs_MSB??.MS/detected_dynamic_spec/{img_name}')[0]
        dyna_data = fits.getdata(dyna_file)
        four_sigma_mask = np.abs(dyna_data) > 4
        six_sigma_mask = np.abs(dyna_data) > 6

        six_sigma_coords = np.argwhere(six_sigma_mask)

        # Initialize a list to store bounding boxes
        bounding_boxes = []

        for coord in six_sigma_coords:
            coord_tuple = tuple(coord)
            source_region, bbox = grow_region(dyna_data, four_sigma_mask, [coord_tuple])
            bounding_boxes.append(bbox)

        # Remove duplicates from bounding_boxes
        unique_bounding_boxes = list(set(bounding_boxes))

        # we might have multiple detections within one dynamic spectrum
        for i in range(len(unique_bounding_boxes)):
            min_freq = unique_bounding_boxes[i][0]
            max_freq = unique_bounding_boxes[i][1]
            min_time = unique_bounding_boxes[i][2]
            max_time = unique_bounding_boxes[i][3]

            min_SB = min_freq // bin_per_SB + SB_min
            max_SB = max_freq // bin_per_SB + SB_min

            # now we find the calibrator

            parts = exo_dir.split("_")
            start_time = parts[0] + "_" + parts[1]
            end_time = parts[2] + "_" + parts[3]
            year = parts[0][:4]
            month = parts[0][4:6]

            base_cal_dir = os.path.join(preprocess_dir, year, month)
            pre_target_dir = os.path.join(preprocess_dir, year, month, exo_dir)
            post_target_dir = os.path.join(postprocess_dir, exo_dir)

            potential_dirs = [d for d in os.listdir(postprocess_dir) if any(cal in d for cal in CALIBRATORS)]
    
            valid_cal_dirs = []
            for dir in potential_dirs:
                parts = dir.split("_")
                dir_start_time = parts[0] + "_" + parts[1]
                dir_end_time = parts[2] + "_" + parts[3]
                if dir_start_time == end_time or dir_end_time == start_time:
                    valid_cal_dirs.append(dir)

            cal_dir = valid_cal_dirs[0]

            # copy data into calibrator directory and exo directory
            for SB in range(min_SB, max_SB+1):
                # copy data into calibrator directory
                cmd = f"rsync -av --progress {base_cal_dir}/{cal_dir}/L1/SB{SB:03d}.MS {postprocess_dir}/{cal_dir}/"
                subprocess.run(cmd, shell=True, check=True)

                # copy data into exo directory
                cmd = f"rsync -av --progress {pre_target_dir}/L1/SB{SB:03d}.MS {post_target_dir}/"
                subprocess.run(cmd, shell=True, check=True)
        



    else:
        continue
