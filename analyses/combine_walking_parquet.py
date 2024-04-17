#!/usr/bin/env ipython

import os
import pandas as pd

os.chdir('/home/lili/data/tuthill/summaries/walking')

fnames = [
    'evyn--Berlin-WT.pq',
    'sarah--rv1-Berlin-WT.pq',
    'sarah--rv3-Berlin-WT.pq',
    'sarah--rv4-Berlin-WT.pq',
    'sarah--rv10-Berlin-WT.pq', # added for v2 
    'sarah--rv14-Berlin-WT.pq',
    'grant--rv15-Berlin-WT.pq',
    'grant--rv16-Berlin-WT.pq',
]

filter_flies = {
    'sarah--rv14-Berlin-WT.pq': [
        '8.18.22 Fly 2_0', '8.19.22 Fly 1_0', '8.19.22 Fly 2_0', '8.19.22 Fly 3_0'
    ],
    'grant--rv15-Berlin-WT.pq': [
        '9.2.22 Fly 3_0', '9.7.22 Fly 2_0', '9.7.22 Fly 3_0',
        '9.7.22 Fly 4_0', '9.9.22 Fly 1_0', '9.9.22 Fly 2_0',
        '9.9.22 Fly 3_0', '9.9.22 Fly 4_0', '9.9.22 Fly 5_0'
    ]
}

ds = []
for fname in fnames:
    print(fname)
    d = pd.read_parquet(fname)
    if fname in filter_flies:
        flies = set(filter_flies[fname])
        d = d.loc[[x in flies for x in d['flyid']]]
    ds.append(d)

data = pd.concat(ds, join='inner', ignore_index=True, copy=False)

data.to_parquet('grant_sarah_evyn_combined_WT_walking_rawang_2.pq', compression='snappy')
