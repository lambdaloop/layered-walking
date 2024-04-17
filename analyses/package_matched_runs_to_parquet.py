#!/usr/bin/env ipython

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

# load pickle files
# convert to 3D poses
# save them back out


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
        deriv = signal.savgol_filter(ang, 5, 2, deriv=1, axis=0)
        accel = signal.savgol_filter(deriv, 5, 2, deriv=1, axis=0)
        conditions.append(cond)
        angles.append(ang)
        derivs.append(deriv)
        accels.append(accel)
        phasesTG.append(d['phaseTG'][i])
        

fname = '/home/lili/data/tuthill/models/models_sls/data_subang_5.pickle'
wd = WalkingData(fname)


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



ix_bout = 0

dfs = []

for ix_bout in trange(len(conditions), ncols=70):
    cond = conditions[ix_bout]
    pose = angles_to_pose_names(angles[ix_bout], angle_names)
    df = pose_to_df(pose)
    df['name'] = 'simulated_fly_{}_b{}'.format(
        format_speed(cond['context']), cond['offset'])
    df['ks'] = ks_angs(angles[ix_bout])
    dfs.append(df)

for ix_bout in trange(len(conditions), ncols=70):
    cond = conditions[ix_bout]
    ww = wd.get_bout(cond['context'], offset=cond['offset'])
    pose = bout_to_pose(ww)[:500]
    df = pose_to_df(pose)
    df['name'] = 'real_fly_{}_b{}'.format(
        format_speed(cond['context']), cond['offset'])
    df['ks'] = ks_bout(ww)[:500]
    dfs.append(df)

outname = '../vids-npz/video_poses_matched.pq'
data = pd.concat(dfs)
data.to_parquet(outname)
