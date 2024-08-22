import os
import shutil
import concurrent.futures
import logging

# Set up logging to both console and file
log_filename = "compression_log.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])

preprocess_dir = "/databf/nenufar-nri/LT02/"
postprocess_dir = "/data/xzhang/exo_img/"

# Sort the list of directories before processing
postprocess_list = sorted([d for d in os.listdir(postprocess_dir)])

def process_observation(obs):
    try:
        parts = obs.split("_")
        year = parts[0][:4]
        month = parts[0][4:6]

        pre_obs_dir = os.path.join(preprocess_dir, year, month, obs)

        if not os.path.isdir(os.path.join(pre_obs_dir, "L2")):
            os.mkdir(os.path.join(pre_obs_dir, "L2"))
        
        tar_gz_path = os.path.join(pre_obs_dir, "L2", obs + ".tar.gz")
        
        if not os.path.exists(tar_gz_path):
            logging.info(f"Compressing {obs} to {tar_gz_path}...")
            try:
                shutil.make_archive(os.path.join(pre_obs_dir, "L2", obs), 'gztar', postprocess_dir, obs)
                logging.info(f"Finished compressing {obs}")
            except FileNotFoundError as fnf_error:
                logging.error(f"File not found during compression: {fnf_error}")
        else:
            logging.info(f"{tar_gz_path} already exists, skipping compression.")
    
    except Exception as e:
        logging.error(f"Error processing {obs}: {e}")

# Increase the number of threads to 32
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(process_observation, postprocess_list)