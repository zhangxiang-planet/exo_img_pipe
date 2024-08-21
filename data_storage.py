import subprocess
import os, glob

# This script is used to store the data in /databf

# First, compare the postprocessing directory with the data directory

preprocess_dir = "/databf/nenufar-nri/LT02/"
postprocess_dir = "/data/xzhang/exo_img/"

# Get the list of the data in the postprocessing directory
postprocess_list = [d for d in os.listdir(postprocess_dir)]

for obs in postprocess_list:
    parts = obs.split("_")
    start_time = parts[0] + "_" + parts[1]
    end_time = parts[2] + "_" + parts[3]
    year = parts[0][:4]
    month = parts[0][4:6]

    pre_obs_dir = os.path.join(preprocess_dir, year, month, obs)

    # Is there an "L2" folder in pre_obs_dir?
    if os.path.isdir(os.path.join(pre_obs_dir, "L2")):
        # If there is, continue
        continue
    else:
        # make the L2 folder
        # print the name of the observation
        print(obs)
        os.mkdir(os.path.join(pre_obs_dir, "L2"))
        # make a tarball of the postprocessing data
        subprocess.call(["tar", "-cvzf", os.path.join(pre_obs_dir, "L2", obs + ".tar.gz"), "-C", postprocess_dir, obs])
