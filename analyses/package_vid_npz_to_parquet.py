#!/usr/bin/env ipython

import numpy as np
import pandas as pd
from glob import glob
import os

files = sorted(glob('../vids-npz/*.npz'))

legs = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']

def pose_to_df(pose):
    columns = {}
    for ileg, leg in enumerate(legs):
        for ia, a in enumerate('ABCDE'):
            for ix, x in enumerate('xyz'):
                name = leg + a + '_' + x
                columns[name] = pose[:, ileg, ia, ix]
    return pd.DataFrame(columns)

dfs = []

for fname in files:
    pose = np.load(fname)['pose']
    basename = os.path.splitext(os.path.basename(fname))[0]
    df = pose_to_df(pose)
    df['name'] = basename
    dfs.append(df)

outname = '../vids-npz/video_poses.pq'
data = pd.concat(dfs)
data.to_parquet(outname)
