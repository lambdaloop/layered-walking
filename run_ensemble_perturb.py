#!/usr/bin/env ipython

import pickle
import numpy as np
from model_functions import MLPScaledXY
from collections import Counter
from scipy.special import expit, logit
import matplotlib.pyplot as plt
import os
from model_functions import get_model_input, run_model
from tqdm import tqdm, trange
import seaborn as sns
import sys

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, ExpSineSquared, WhiteKernel


import tensorflow as tf

# memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# with open('models/full_model_withrot.pickle', 'rb') as f:
# with open('models/full_model_corrected_extreme.pickle', 'rb') as f:
# with open('models/byfly2/6.15.20 Fly 4_0.pickle', 'rb') as f:
# with open('models_ensemble/model_p25_a10_e10_n100_i90.pickle', 'rb') as f:

# basename = 'model_p25_a10_e10_n100'
# basename = 'model_p75_a10_e10_n100'
# basename = 'model_p25_a10_e10_n100_sp00'

basename = sys.argv[1]

kernel = RBF(1.0, length_scale_bounds="fixed") # + RBF(0.5, length_scale_bounds='fixed')
xvals = np.linspace(-np.pi, np.pi)

def phase_align(ang):
    dm = ang - np.median(ang, axis=0)
    dm = dm / np.std(dm, axis=0)
    phase = np.arctan2(-dm[:,1], dm[:,0])
    gauss = GaussianProcessRegressor(kernel).fit(phase[:,None], ang)
    yvals_model = gauss.predict(xvals[:,None])
    return yvals_model

ball_radius = 0.478
fictrac_fps = 30.0
ratio = ball_radius * fictrac_fps * 10
fake_speed = 10


models_data = []
for inum in trange(100):
    fname = 'models_ensemble/{}_i{}.pickle'.format(basename, inum)
    try:
        with open(fname, 'rb') as f:
            model = pickle.load(f)
    except:
        continue
    models_data.append(model)

(xy_s_test, xy_w_test), bnums_test = models_data[0]['test']
bout_number, _ = Counter(bnums_test).most_common()[0]
offset = 0
n_pred = 600

# props: claw flex, claw ext, club, hook flex, hook ext

outs = dict()
for fake_speed in [4, 8, 12, 16]:
  subout = dict()
  for ix_perturb in range(5):
      Ls = []
      for add in tqdm(np.arange(-1.2, 1.3, 0.4), ncols=70):
          # px = {'start': 200, 'end': 400, 'add': add, 'ix': ix_perturb}
          px = {'start': 200, 'end': 300, 'add': add, 'ix': ix_perturb}
          L = []
          for ix_model, model in enumerate(models_data):
              stuff = get_model_input(model, bout_number, offset, n_pred, fake_speed / ratio)
              n_pred, init, context, prop_params, models, params = stuff
              perturbs = [px]
              out = run_model(*stuff, perturbations=perturbs)
              out['perturbs'] = perturbs
              L.append(out)
          Ls.append(L)
      subout[ix_perturb] = Ls
  outs[fake_speed] = subout

outname = 'runs/{}_perturbs_speeds_100_shortperturb.pickle'.format(basename)
with open(outname, 'wb') as f:
    pickle.dump(outs, f)
