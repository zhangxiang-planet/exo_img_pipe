import os
import shutil
import concurrent.futures

preprocess_dir = "/databf/nenufar-nri/LT02/"
postprocess_dir = "/data/xzhang/exo_img/"

postprocess_list = [d for d in os.listdir(postprocess_dir)]

def process_observation(obs):
    parts = obs.split("_")
    start_time = parts[0] + "_" + parts[1]
    end_time = parts[2] + "_" + parts[3]
    year = parts[0][:4]
    month = parts[0][4:6]

    pre_obs_dir = os.path.join(preprocess_dir, year, month, obs)

    if not os.path.isdir(os.path.join(pre_obs_dir, "L2")):
        os.mkdir(os.path.join(pre_obs_dir, "L2"))
        if not os.path.exists(os.path.join(pre_obs_dir, "L2", obs + ".tar.gz")):
            shutil.make_archive(os.path.join(pre_obs_dir, "L2", obs), 'gztar', postprocess_dir, obs)

with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(process_observation, postprocess_list)