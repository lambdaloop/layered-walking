#!/usr/bin/env ipython

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from glob import glob
import os

import sys
project_path = "/home/lili/research/tuthill/layered-walking"
data_path = '/home/lili/data/tuthill/models/sls_runs'
sys.path.append(project_path)
from tools.angle_functions import anglesTG as angle_names_1leg
from tools.angle_functions import legs
from tools.angle_functions import make_fly_video, angles_to_pose_names
from tools.trajgen_tools import WalkingData

from tqdm import tqdm, trange
import pickle

fname_pat = os.path.join(data_path, 'match_v9_*.pkl')
fnames = sorted(glob(fname_pat))

conditions = []
angles = []
derivs = []
accels = []
phasesTG = []

for fname in tqdm(fnames, ncols=70):
    # d = np.load(fname, allow_pickle=True)
    with open(fname, 'rb') as f:
        d = pickle.load(f)

    angle_names = d['angleNames'][0]

    for i, cond in enumerate(d['conditions']):
        ang = d['angle'][i]
        conditions.append(cond)
        angles.append(ang)
        phasesTG.append(d['phaseTG'][i])
        

fname = '/home/lili/data/tuthill/models/models_sls/data_subang_5.pickle'
wd = WalkingData(fname)

video_speeds = [
    ((8, 0, 0), 7), # 2
    ((10, 0, 0), 2),
    ((12, 0, 0), 6),
    ((14, 0, 0), 3),
    ((12, -8, 0), 4),
    ((12, 8, 0), 6),
    ((12, 0, -4), 2),
    ((12, 0, 4), 7)
]

legs = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']
def pose_to_df(pose):
    columns = {}
    for ileg, leg in enumerate(legs):
        for ia, a in enumerate('ABCDE'):
            for ix, x in enumerate('xyz'):
                name = leg + a + '_' + x
                columns[name] = pose[:, ileg, ia, ix]
    return pd.DataFrame(columns)

def bout_to_pose(bout):
    out = []
    angnames = []
    for leg in legs:
        ang = bout['angles'][leg]
        out.append(ang)
        angnames.extend(wd._angle_names[leg])
    ang = np.hstack(out)
    pose = angles_to_pose_names(ang, angnames)
    return pose

def format_speed(speed):
    return '_'.join([str(x) for x in speed])

def ks_angs(angs):
    pdfs = np.full(len(angs), -2.5)
    angs_sc = np.hstack([np.sin(np.deg2rad(angs)),
                      np.cos(np.deg2rad(angs))])
    c = np.all(np.isfinite(angs_sc), axis=1)
    angs_sc = angs_sc[c]
    pangs = wd.data['pca'].transform(angs_sc)
    pdfs[c] = wd.data['kde'].logpdf(pangs.T)
    return pdfs

def ks_bout(bout):
    angs = np.hstack([bout['angles'][leg] for leg in legs])
    return ks_angs(angs)


for ix_bout in range(len(conditions)):
    cond = conditions[ix_bout]
    x = (tuple(cond['context']), cond['offset'])
    if x not in video_speeds: continue
    pose = angles_to_pose_names(angles[ix_bout], angle_names)
    ks_mean = np.mean(ks_angs(angles[ix_bout]))
    print('simulated ', cond['context'], ks_mean)
    make_fly_video(pose, '../vids/simulated_fly_{}.mp4'.format(format_speed(cond['context'])),
                   progress=False)

for ix_bout in range(len(conditions)):
    cond = conditions[ix_bout]
    x = (tuple(cond['context']), cond['offset'])
    if x not in video_speeds: continue
    ww = wd.get_bout(cond['context'], offset=cond['offset'])
    pose = bout_to_pose(ww)[:500]
    ks_mean = np.mean(ks_bout(ww)[:500])
    print('real ', cond['context'], ks_mean)
    make_fly_video(pose, '../vids/real_fly_{}.mp4'.format(format_speed(cond['context'])),
                   progress=False)
