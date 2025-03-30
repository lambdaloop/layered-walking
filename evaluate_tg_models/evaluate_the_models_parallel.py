from glob import glob
import os
from evaluate_model_functions import evaluate_model
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import warnings

file_list = []

# directory = '/home/lili/data/tuthill/models/models_sls_sweep'
# file_list += sorted(glob(os.path.join(directory, '*.pickle')))
# file_list += sorted(glob(os.path.join(directory, '*.pkl')))

# directory = '/home/lili/data/tuthill/models/models_sls_sweep_moreiters'
# file_list += sorted(glob(os.path.join(directory, '*.pickle')))
# file_list += sorted(glob(os.path.join(directory, '*.pkl')))

directory = '/home/lili/data/tuthill/models/models_sls_sweep_v3'
file_list += sorted(glob(os.path.join(directory, '*.pickle')))
file_list += sorted(glob(os.path.join(directory, '*.pkl')))

default_params = {
    'hidden_dim': np.nan,
    'dropout_rate': np.nan,
    'phase_noise': np.nan,
    'n_pred_btt': np.nan
}

num_processes = 20

def process_file(filepath):
    basename = os.path.basename(filepath)
    dirname = os.path.basename(os.path.dirname(filepath))
    # print()
    # print(basename)
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
        params = model.get('train_params', default_params)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        d = evaluate_model(filepath, progress=False)
    d['model'] = dirname + "/" + basename
    for k, v in params.items():
        d[k] = v
    return d

with Pool(num_processes) as pool:
    ds = list(tqdm(pool.imap(process_file, file_list), total=len(file_list), ncols=80))

dout = pd.concat(ds)

outpath = os.path.join(directory, 'models_errors_speedhack_v2.csv')
dout.to_csv(outpath, index=False)
