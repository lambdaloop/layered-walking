from glob import glob
import os
from evaluate_model_functions import evaluate_model
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm

directory = '/home/lili/data/tuthill/models/models_sls_sweep'
file_list = sorted(glob(os.path.join(directory, '*.pickle')))
file_list += sorted(glob(os.path.join(directory, '*.pkl')))

default_params = {
    'hidden_dim': np.nan,
    'dropout_rate': np.nan,
    'phase_noise': np.nan,
    'n_pred_btt': np.nan
}

ds = []
for filepath in tqdm(file_list, ncols=80):
    basename = os.path.basename(filepath)
    print()
    print(basename)
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
        params = model.get('train_params', default_params)
    d = evaluate_model(filepath, progress=False)
    d['model'] = basename
    for k, v in params.items():
        d[k] = v
    ds.append(d)

dout = pd.concat(ds)

outpath = os.path.join(directory, 'models_errors.csv')
dout.to_csv(outpath, index=False)
