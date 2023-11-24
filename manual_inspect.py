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

singularity_command = f"singularity exec -B/data/$USER {singularity_file}"

######################

def split_into_chunks(lst, chunk_size):
    """Split a list into chunks of specified size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


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
img_list_i = glob.glob(watch_dir + "*_png_i/*.png")
img_list_v = glob.glob(watch_dir + "*_png_v/*.png")
img_list = img_list_i + img_list_v
img_list.sort()

# loop over the images
for img in img_list:
    suffix = 'i' if '_png_i/' in img else 'v'
    # do we have a corresponding fits file?
    bound_file = img.replace(".png", ".bound_box.txt")
    if not os.path.exists(bound_file):
        exo_dir = img.split("/")[-2].replace(f"_png_{suffix}", "")

        # get the calibrator name
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

        for cal in CALIBRATORS:
            if cal in cal_dir:
                calibrator = cal

                
        img_name = img.split("/")[-1].replace(".png", "")
        dyna_file = glob.glob(f'{postprocess_dir}{exo_dir}/dynamic_spec_DynSpecs_MSB??.MS/detected_dynamic_spec_{suffix}/{img_name}')[0]
        dyna_data = fits.getdata(dyna_file)
        # we need the shape of the dynamic spectrum
        num_chan, num_ts = dyna_data.shape
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

        np.savetxt(f'{watch_dir}{exo_dir}_png_{suffix}/{img_name}.bound_box.txt', unique_bounding_boxes, fmt='%d')

        # we might have multiple detections within one dynamic spectrum
        for i in range(len(unique_bounding_boxes)):
            min_freq = unique_bounding_boxes[i][0]
            max_freq = unique_bounding_boxes[i][1]
            min_time = unique_bounding_boxes[i][2]
            max_time = unique_bounding_boxes[i][3]

            # we skip this bounding box if it is too narrow in time or frequency
            if max_freq - min_freq < 3 or max_time - min_time < 3:
                continue

            # find the actual min SB, which is the bigger one in SB_min and the min of SBs within {base_cal_dir}/{cal_dir}/L1/
            cali_SBs = glob.glob(base_cal_dir + "/" + cal_dir + "/L1/SB*.MS")
            cali_SBs.sort()
            cali_min = cali_SBs[0].split("/")[-1].split(".")[0].split("SB")[-1]
            SB_min = max(SB_min, int(cali_min))

            min_SB = min_freq // bin_per_SB + SB_min
            max_SB = max_freq // bin_per_SB + SB_min

            num_SB = max_SB - min_SB + 1

            num_time = max_time - min_time + 1

            # copy data into calibrator directory and exo directory
            for SB in range(min_SB, max_SB+1):
                # copy data into calibrator directory
                cmd = f"rsync -av --progress {base_cal_dir}/{cal_dir}/L1/SB{SB:03d}.MS {postprocess_dir}/{cal_dir}/"
                subprocess.run(cmd, shell=True, check=True)

                # copy data into exo directory
                cmd = f"rsync -av --progress {pre_target_dir}/L1/SB{SB:03d}.MS {post_target_dir}/"
                subprocess.run(cmd, shell=True, check=True)

            cali_SB = glob.glob(postprocess_dir + cal_dir + '/SB*.MS')
            cali_SB.sort()

            target_SB = glob.glob(postprocess_dir + exo_dir + '/SB*.MS')
            target_SB.sort()

            # Splitting cali_SB and target_SB into chunks
            cali_SB_chunks = list(split_into_chunks(cali_SB, chunk_num))
            target_SB_chunks = list(split_into_chunks(target_SB, chunk_num))

            processed_cali_SBs = []
            processed_target_SBs = []

            for chunk_idx, (cali_SB_chunk, target_SB_chunk) in enumerate(zip(cali_SB_chunks, target_SB_chunks)):
                cali_SB_str = ",".join(cali_SB_chunk)
                target_SB_str = ",".join(target_SB_chunk)

                # Names for the Multi-Subband (MSB) data
                MSB_cali = f"{postprocess_dir}/{cal_dir}/MSB_candidate_{i}_chunk_{chunk_idx}.MS"
                MSB_target = f"{postprocess_dir}{exo_dir}/MSB_candidate_{i}_chunk_{chunk_idx}.MS"

                num_SB_chunk = len(cali_SB_chunk)

                processed_cali_SBs.append(MSB_cali)
                processed_target_SBs.append(MSB_target)


                ############################

                # Construct the command string with the msin argument and the msout argument
                cmd_flagchan = f"DP3 {pipe_dir}/templates/DPPP-flagchan.parset msin=[{cali_SB_str}] msout={MSB_cali} avg.freqstep={ave_chan}"
                subprocess.run(cmd_flagchan, shell=True, check=True)

                cmd_flagMA = f"DP3 {postprocess_dir}/{cal_dir}/DPPP-flagant.parset msin={MSB_cali}"
                subprocess.run(cmd_flagMA, shell=True, check=True)

                # Construct the command string with the msin argument and the msout argument
                cmd_aoflagger = f"DP3 {pipe_dir}/templates/DPPP-aoflagger.parset msin={MSB_cali} flag.strategy={pipe_dir}/templates/Nenufar64C1S.lua"
                subprocess.run(cmd_aoflagger, shell=True, check=True)

                with open(f'{pipe_dir}/templates/bad_MA.toml', 'r') as template_file:
                    template_content = template_file.read()

                cali_model = f'{pipe_dir}/cal_models/{calibrator}_lcs.skymodel'

                modified_content = template_content.replace('CALI_MODEL', cali_model)
                modified_content = modified_content.replace('CHAN_PER_SB', str(num_SB))

                # Write the modified content to a new file
                with open(f'{postprocess_dir}/{cal_dir}/cali_candidate_{i}.toml', 'w') as cali_file:
                    cali_file.write(modified_content)

                cmd_cali = f"calpipe {postprocess_dir}/{cal_dir}/cali_candidate_{i}.toml {MSB_cali}"
                subprocess.run(cmd_cali, shell=True, check=True)

                # Remove the table files so they don't take up too much space!
                cmd_remo_table = f"rm -rf {MSB_cali}/table.* {MSB_cali}/pre_cal_flags.h5"
                subprocess.run(cmd_remo_table, shell=True, check=True)

                ############################

                cmd_flagchan = f"DP3 {pipe_dir}/templates/DPPP-flagchan.parset msin=[{target_SB_str}] msout={MSB_target} avg.freqstep={ave_chan}"
                subprocess.run(cmd_flagchan, shell=True, check=True)

                if os.path.exists(f'{postprocess_dir}/{exo_dir}/DPPP-removeant.parset'):
                    cmd_removeMA = f"DP3 {postprocess_dir}/{exo_dir}/DPPP-removeant.parset msin={MSB_target} msout={MSB_target}B"
                    subprocess.run(cmd_removeMA, shell=True, check=True)

                    cmd_remo_MSB = f"rm -rf {MSB_target}"
                    subprocess.run(cmd_remo_MSB, shell=True, check=True)
                    # rename the new MSB
                    cmd_rename_MSB = f"mv {MSB_target}B {MSB_target}"
                    subprocess.run(cmd_rename_MSB, shell=True, check=True)

                cmd_flagMA = f"DP3 {postprocess_dir}/{exo_dir}/DPPP-flagant.parset msin={MSB_target}"
                subprocess.run(cmd_flagMA, shell=True, check=True)

                cmd_aoflagger = f"DP3 {pipe_dir}/templates/DPPP-aoflagger.parset msin={MSB_target} flag.strategy={pipe_dir}/templates/Nenufar64C1S.lua"
                subprocess.run(cmd_aoflagger, shell=True, check=True)

                cmd_copy_solution = f'cp {MSB_cali}/instrument_ddecal.h5 {MSB_target}/instrument_dical.h5'
                subprocess.run(cmd_copy_solution, shell=True, check=True)

                cmd_apply_solution = f'calpipe {pipe_dir}/templates/cali_tran.toml {MSB_target}'
                subprocess.run(cmd_apply_solution, shell=True, check=True)

                # second round of aoflagger
                cmd_aoflagger = f"DP3 {pipe_dir}/templates/DPPP-aoflagger.parset msin={MSB_target} msin.datacolumn=DI_DATA flag.strategy={pipe_dir}/templates/Nenufar64C1S.lua"
                subprocess.run(cmd_aoflagger, shell=True, check=True)

                ############################
                # Don't forget to remove the A team sources
                cmd_ddf = (
                    f'DDF.py --Data-MS {MSB_target} --Data-ColName DI_DATA --Output-Name {postprocess_dir}{exo_dir}/MSB_candidate_{i}_chunk_{chunk_idx}_Image_DI '
                    f'--Image-Cell 60 --Image-NPix 2400 --Output-Mode Clean --Facets-NFacets 5 --Parallel-NCPU 96 --Freq-NBand {num_SB_chunk} --Freq-NDegridBand 0 '
                    '--Selection-UVRangeKm [0.067,1000] --Comp-GridDecorr 0.0001 --Comp-DegridDecorr 0.0001 --Deconv-Mode HMP --Deconv-MaxMajorIter 20 '
                    '--Mask-Auto 1 --Mask-SigTh 4 --Deconv-AllowNegative 0 --Deconv-RMSFactor 4 --Output-Also all'
                )
                combined_ddf = f"{singularity_command} {cmd_ddf}"
                subprocess.run(combined_ddf, shell=True, check=True)

                cmd_kms = (
                    f'kMS.py --MSName {MSB_target} --SolverType CohJones --PolMode IFull --BaseImageName {postprocess_dir}{exo_dir}/MSB_candidate_{i}_chunk_{chunk_idx}_Image_DI '
                    f'--dt 2 --InCol DI_DATA --OutCol SUB_DATA --SolsDir={postprocess_dir}/{exo_dir}/SOLSDIR --NodesFile Single --DDFCacheDir={postprocess_dir}{exo_dir}/ --NChanPredictPerMS {num_SB_chunk} --NChanSols {num_SB_chunk} '
                    '--OutSolsName DD1 --UVMinMax 0.067,1000 --AppendCalSource All --FreePredictGainColName KMS_SUB:data-ATeam'
                )
                combined_kms = f"{singularity_command} {cmd_kms}"
                subprocess.run(combined_kms, shell=True, check=True)

            processed_cali_SB_str = ",".join(processed_cali_SBs)
            processed_target_SB_str = ",".join(processed_target_SBs)


            ###########################

            # now we use wsclean to image the time range
            # ideally, we also image the time range before the burst and after the burst, but we need to know if they fit into the observing window

            # first, we image the burst
            cmd_burst_img = (f'wsclean -pol I,Q,U,V -weight briggs 0 -data-column KMS_SUB -minuv-l 0 -maxuv-l 1000 ' 
                             f'-scale 1amin -size 2400 2400 -make-psf -niter 0 -auto-mask 6 -auto-threshold 5 -mgain 0.6 '
                             f'-local-rms -join-polarizations -multiscale -no-negative -no-update-model-required -no-dirty '
                             f'-interval {min_time} {max_time+1} -name {postprocess_dir}/{exo_dir}/MSB_{min_freq}_{max_freq}_{min_time}_{max_time} {postprocess_dir}/{exo_dir}/MSB_candidate_{i}_chunk_?.MS')
            subprocess.run(cmd_burst_img, shell=True, check=True)

            if min_time - num_time > 0:
                # we can image the time range before the burst
                cmd_pre_img = (f'wsclean -pol I,Q,U,V -weight briggs 0 -data-column KMS_SUB -minuv-l 0 -maxuv-l 1000 ' 
                               f'-scale 1amin -size 2400 2400 -make-psf -niter 0 -auto-mask 6 -auto-threshold 5 -mgain 0.6 '
                               f'-local-rms -join-polarizations -multiscale -no-negative -no-update-model-required -no-dirty '
                               f'-interval {min_time-num_time} {min_time} -name {postprocess_dir}/{exo_dir}/MSB_{min_freq}_{max_freq}_{min_time}_{max_time}_pre {postprocess_dir}/{exo_dir}/MSB_candidate_{i}_chunk_?.MS')
                subprocess.run(cmd_pre_img, shell=True, check=True)

            if max_time + num_time < num_ts:
                # we can image the time range after the burst
                cmd_post_img = (f'wsclean -pol I,Q,U,V -weight briggs 0 -data-column KMS_SUB -minuv-l 0 -maxuv-l 1000 ' 
                                f'-scale 1amin -size 2400 2400 -make-psf -niter 0 -auto-mask 6 -auto-threshold 5 -mgain 0.6 '
                                f'-local-rms -join-polarizations -multiscale -no-negative -no-update-model-required -no-dirty '
                                f'-interval {max_time+1} {max_time+num_time+1} -name {postprocess_dir}/{exo_dir}/MSB_{min_freq}_{max_freq}_{min_time}_{max_time}_post {postprocess_dir}/{exo_dir}/MSB_candidate_{i}_chunk_?.MS')
                subprocess.run(cmd_post_img, shell=True, check=True)

        cmd_remo_SB = f"rm -rf {postprocess_dir}/{exo_dir}/SB*.MS"
        subprocess.run(cmd_remo_SB, shell=True, check=True)

        cmd_remo_SB = f"rm -rf {postprocess_dir}/{cal_dir}/SB*.MS"
        subprocess.run(cmd_remo_SB, shell=True, check=True)

    else:
        continue
