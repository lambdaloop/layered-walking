#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')

from tools.trajgen_tools import WalkingData, TrajectoryGenerator
from tools.angle_functions import make_fly_video, angles_to_pose_names
from evaluate_model_functions import simulate_bout
import numpy as np
import os
from glob import glob
from tqdm import tqdm, trange
from multiprocessing import Pool
import warnings
import sys

# data_filename = '/home/lili/data/tuthill/models/models_sls/data_subang_2.pickle'
data_filename = '/home/lili/data/tuthill/models/models_sls/data_subang_5.pickle'
# tg_filename = '/home/lili/data/tuthill/models/models_sls/walk_sls_legs_subang_9.pickle'

# directory = '/home/lili/data/tuthill/models/models_sls_sweep'
# file_list = sorted(glob(os.path.join(directory, '*.pickle')))
# file_list += sorted(glob(os.path.join(directory, '*.pkl')))

# directory = '/home/lili/data/tuthill/models/models_sls_sweep_moreiters'
# file_list += sorted(glob(os.path.join(directory, '*.pickle')))
# file_list += sorted(glob(os.path.join(directory, '*.pkl')))

directory = '/home/lili/data/tuthill/models/models_sls_sweep_v3'
file_list = sorted(glob(os.path.join(directory, '*.pickle')))
file_list += sorted(glob(os.path.join(directory, '*.pkl')))

# tg_filename = '/home/lili/data/tuthill/models/models_sls_sweep/model_hd032_dr005_pn020_np000.pkl'

outfolder = '/home/lili/data/tuthill/models/vids_sls_sweep_v3_speedhack'

legs = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']
n_pred = 500
n_real = 6
num_processes = 18

speeds = [[8, 0, 0], [10, 0, 0], [12, 0, 0], [14, 0, 0],
          [12, 0, -4], [12, 0, 4], [12, -8, 0], [12, 8, 0]]


def bout_to_pose(bout):
    angles = np.hstack([bout['angles'][leg] for leg in legs])
    angle_names = np.hstack(all_angles_main)
    pose = angles_to_pose_names(angles, angle_names)
    return pose

## This isn't very clean
## You could refactor by placing speed as an argument within process_file and show_real
## and loop over both speed and filename
## i'm too lazy for this right now though in this single use script, and this works
## bite me

for speed in speeds:
    print(speed)

    speed_folder = '{}_{}_{}'.format(*speed)
    outdir = os.path.join(outfolder, speed_folder)
    os.makedirs(outdir, exist_ok=True)

    wd = WalkingData(data_filename)
    all_angles_main = [wd.data[leg]['angle_names'] for leg in legs]
    bout_real = wd.get_bout(speed, offset=0, min_bout_length=n_pred)

    # force the speed to be constant for simulation here
    bout_real['contexts'] = np.tile(speed, (n_pred, 1))

    def process_file(tg_filename):
        TG = [TrajectoryGenerator(tg_filename, leg, n_pred) for leg in legs]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            bout_sim = simulate_bout(TG, bout_real)

        model_name = os.path.splitext(os.path.basename(tg_filename))[0]
        outpath = os.path.join(outdir, model_name + '.mp4')

        pose = bout_to_pose(bout_sim)
        make_fly_video(pose, outpath, progress=False)

    def show_real(offset):
        bout = wd.get_bout(speed, offset=offset, min_bout_length=n_pred)
        pose = bout_to_pose(bout)
        outpath = os.path.join(outdir, 'real_{}.mp4'.format(offset))
        make_fly_video(pose, outpath, progress=False)

    with Pool(num_processes) as pool:
        list(tqdm(pool.imap(process_file, file_list), total=len(file_list), ncols=80))

    with Pool(num_processes) as pool:
        list(tqdm(pool.imap(show_real, range(n_real)), total=n_real, ncols=80))
