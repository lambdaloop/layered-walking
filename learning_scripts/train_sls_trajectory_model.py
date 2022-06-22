#!/usr/bin/env ipython

import pandas as pd
import numpy as np

from tqdm import tqdm, trange
from scipy import stats, signal
import os
from collections import Counter
import pickle
import time
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from model_functions import get_props_by_bouts, get_sw_xy, get_period, get_speed
from model_functions import summarize, wrap_array
from model_functions import MLPScaledXY, num_vars

# fly = '6.15.20 Fly 4_0'
# fly = sys.argv[1]

print("TensorFlow version: {}".format(tf.__version__))

# prefix = '/home/pierre/data/tuthill/summaries/v3/processed'
# prefix = '/data/users/pierre/gdrive/Tuthill Lab Shared/shared_data/2021-06-07-karashchuk-flyangles'
prefix = '/home/pierre/data/tuthill/summaries/v3-b3/lines'
# prefix = '/home/pierre/Downloads/test'
fnames = [
  "evyn--Berlin-WT.pq",  "sarah--rv1-Berlin-WT.pq",
  # "sarah--rv4-Berlin-WT.pq",
  # "sarah--rv3-Berlin-WT.pq",  "sarah--rv10-Berlin-WT.pq"
]
# data = pd.read_parquet(os.path.join(prefix, 'evyn-sarah-Berlin-WT-phase.pq'))
# data = pd.read_parquet(os.path.join(prefix, 'sarah--rv1-Berlin-WT.pq'))
ds = []
for fname in fnames:
  print(fname)
  d = pd.read_parquet(os.path.join(prefix, fname))
  ds.append(d)
data = pd.concat(ds)


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


FPS = 300.0

legs = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']
items = {
    'A_flex': 'coxa abduction',
    'A_abduct': 'body-coxa flexion',
    'A_rot': 'coxa rotation',
    'B_flex': 'coxa-femur flexion',
    'B_rot': 'femur rotation',
    'C_flex': 'femur-tibia flexion',
    'C_rot': 'tibia rotation',
    'D_flex': 'tibia-tarsus flexion'
}

name_dict = dict()

names = []
for leg in legs:
    for k,v in items.items():
        name = leg + k
        names.append(name)
        name_dict[name] = leg + ' ' + v
names = np.array(names)

# small_main = ['C_flex']
# subset_main = ['C_flex', 'A_abduct', 'B_flex', 'B_rot', 'A_rot']
subset_main = ['C_flex', 'A_abduct', 'B_flex', 'B_rot']
# full_main = ['A_flex', 'A_abduct', 'A_rot', 'B_flex', 'B_rot', 'C_flex', 'C_rot', 'D_flex']
# full_main = ['A_flex', 'A_abduct', 'A_rot', 'B_flex', 'B_rot', 'C_flex', 'C_rot']
# full_main = ['A_abduct', 'A_flex', 'B_flex', 'C_flex', 'D_flex', 'A_rot', 'B_rot', 'C_rot']

# phases_d = data.loc[:, [leg + '_walking_phase' for leg in legs]]
# check = np.all(np.isfinite(phases_d), axis=1)
check = data['walking_bout_number'].notna().values

angles_main = ['L1C_flex', 'L1A_abduct', 'L1B_flex', 'L1B_rot']

angle_names = [leg + name for leg in legs for name in subset_main]
angle_deriv_names = [leg + name +"_d1" for leg in legs for name in subset_main]
angle_accel_names = [leg + name +"_d2" for leg in legs for name in subset_main]

all_names = angle_names + angle_deriv_names
# all_names = joint_names + joint_deriv_names

angles_raw = np.abs(data.loc[check, angle_names].values)
# angles_deriv = data.loc[check, angle_deriv_names].values
# angles_accel = data.loc[check, angle_accel_names].values
# phases = np.mod(phases_d.loc[check].values, 2*np.pi)

fullfiles = data.loc[check, 'fullfile'].to_numpy()
framenums = data.loc[check, 'fnum'].to_numpy()
flyids = data.loc[check, 'flyid'].to_numpy()

bout_numbers_raw = data.loc[check, 'walking_bout_number'].values.astype('int64')
bout_numbers = flyids + " b" + bout_numbers_raw.astype('str')


# fictrac_vals = data.loc[check, ['fictrac_speed', 'fictrac_rot']].values
fictrac_vals = data.loc[check, [
  'fictrac_speed',
  'fictrac_delta_rot_lab_x',
  'fictrac_delta_rot_lab_y',
  'fictrac_delta_rot_lab_z']].values


angles_deriv = np.zeros(angles_raw.shape)
angles_accel = np.zeros(angles_raw.shape)
phases = np.zeros(angles_raw.shape)
phases_deriv = np.zeros(phases.shape)

sos = signal.butter(1, (0.02, 0.4), 'bandpass', output='sos')

for f in tqdm(np.unique(fullfiles), ncols=70):
  cc = fullfiles == f
  ang = angles_raw[cc]
  angles_deriv[cc] = signal.savgol_filter(ang, 5, 2, deriv=1, axis=0) # * FPS
  angles_accel[cc] = signal.savgol_filter(ang, 5, 2, deriv=2, axis=0) # * FPS * FPS
  ang_f = signal.sosfiltfilt(sos, ang, axis=0)
  phases[cc] = np.mod(np.angle(signal.hilbert(ang_f)), 2*np.pi)
  phases_deriv[cc] = signal.savgol_filter(np.unwrap(phases[cc], axis=0),
                                          5, 2, deriv=1, axis=0, mode='nearest')


def get_context_var(name, c):
    if name == 'fictrac_speed':
        return fictrac_vals[c, 0]
    elif name == 'fictrac_rot':
        return fictrac_vals[c, 1:4].T
    elif name == 'period':
        ix = angle_names.index('L1C_flex')
        return get_period(angles_raw[c, ix])
    elif name == 'frequency':
        ix = angle_names.index('L1C_flex')
        return get_speed(angles_raw[c, ix])
    else:
        raise ValueError("invalid context name: {}".format(name))

def get_context(context_list, c):
    if len(context_list) == 0:
        return np.zeros((np.sum(c), 0))
    L = []
    for name in context_list:
        L.append(get_context_var(name, c))
    context = np.vstack(L).T
    return context

def get_data(good_bouts, params):
    ix = [angle_names.index(name) for name in angles_main]
    c = np.isin(bout_numbers, good_bouts)
    inp = np.hstack([angles_raw[:,ix][c], angles_deriv[:,ix][c]])

    praw = wrap_array(phases[c, 0])
    pderiv = wrap_array(phases_deriv[c, 0])
    context = get_context(params['context'], c)
    fnum = framenums[c]
    fname = fullfiles[c]
    check = (fname[1:] == fname[:-1]) & (fnum[1:]-1 == fnum[:-1])
    bnums = bout_numbers[c][1:][check]

    x_walk = np.hstack([inp, context, np.cos(praw), np.sin(praw)])[:-1]
    y_walk = np.hstack([inp, pderiv])[1:]

    x_walk = x_walk[check].astype('float32')
    y_walk = y_walk[check].astype('float32')
    msx_w = summarize(x_walk)
    msy_w = summarize(y_walk)

    return (x_walk, y_walk, msx_w, msy_w), bnums

def filter_bouts(bnums):
  ix = angle_names.index('L1C_flex')
  good_bouts = []
  for bnum in tqdm(np.unique(bnums), ncols=70):
      # if bnum == 0 or np.isnan(bnum): continue
      # cc = np.isclose(bout_numbers, bnum)
      cc = bout_numbers == bnum
      raw = np.abs(angles_raw[cc, ix])
      # deriv = angles_deriv[cc, ix] / FPS
      low, high = np.percentile(raw, [5, 95])
      # high_deriv = np.percentile(deriv, 95)
      vals = fictrac_vals[cc]
      if not np.all(np.isfinite(vals)):
          continue
      check = np.mean(vals[:,0]) > 0.02
      if check and  high - low > 40 and len(raw) >= 150:
          good_bouts.append(bnum)
  good_bouts = np.array(good_bouts)
  return good_bouts

## get good bouts
# fly = "6.15.20 Fly 4_0"
# fly = "all"
# bnums = np.unique(bout_numbers[flyids == fly])
bnums = np.unique(bout_numbers)

# inp = angles_raw, angles_deriv, bout_numbers, fictrac_vals, angle_names
# good_bouts = filter_bouts(bnums, inp)
good_bouts = filter_bouts(bnums)


np.random.seed(123)
np.random.shuffle(good_bouts)

# params = {'context': ['fictrac_speed', 'fictrac_rot'], 'use_phase': True}
params = {'context': ['fictrac_speed'], 'use_phase': True}

xy_w, bnums = get_data(good_bouts[:-5], params)
xy_w_test, bnums_test = get_data(good_bouts[-5:], params)

print("Data points in training set:", len(xy_w[0]))
print("Data points in test set:", len(xy_w_test[0]))



model_walk = MLPScaledXY(output_dim=xy_w[1].shape[1],
                         hidden_dim=128, dropout_rate=0.05,
                         msx=xy_w[2], msy=xy_w[3])


model_walk(xy_w[0][:2])
print('Walk', num_vars(model_walk))


in_walk = xy_w[0]
in_walk_state = in_walk[:, :-3]
extra_walk = in_walk[:, -3:]
out_walk = xy_w[1]

# prob_perturb = 0.5
# sd_err_deriv = np.std(in_walk[:,-1]) * 0.5
# accel_ratio = 1.5

lr = tf.Variable(1e-3)
opt = tf.keras.optimizers.Adam(learning_rate=lr)

@tf.function
def step_mlp_norm(model_walk, in_walk, out_walk):
  """Performs one optimizer step on a single mini-batch."""
  in_walk = tf.cast(in_walk, 'float32')
  out_walk = tf.cast(out_walk, 'float32')

  N = out_walk.shape[0]

  with tf.GradientTape() as tape:
      pred_walk = model_walk(in_walk, is_training=True)
      error_walk = tf.square(out_walk - pred_walk) / tf.square(model_walk.msy[1])
      loss = tf.reduce_mean(error_walk)

  variables = model_walk.trainable_variables
  grads = tape.gradient(loss, variables)
  opt.apply_gradients(zip(grads, variables))
  return loss


# batch_size = 2500
# n_epochs = 6000
batch_size = 8000
n_epochs = 200
# n_epochs = 1000


t0 = time.time()

for epoch_num in range(n_epochs+1):
    ixs = np.arange(len(in_walk))
    np.random.shuffle(ixs)
    total = 0
    num = 0
    for s in range(0, len(ixs), batch_size):
        c = ixs[s:s+batch_size]
        in_walk_c = np.copy(in_walk[c])
        out_walk_c = np.copy(out_walk[c])
        total += step_mlp_norm(model_walk, in_walk_c, out_walk_c).numpy()
        num += 1
    if epoch_num % 25 == 0:
        t1 = time.time() - t0
        print("Time: {:.2f} Epoch {}: {:.5f}".format(t1, epoch_num, total / num))


model_standard = {
    # 'fly': fly,
    'model_walk': model_walk.get_full(),
    # 'prop_params': (claw_flex_basic, claw_ext_basic, club_squash, hook_flex_squash),
    # 'prop_params': prop_params,
    'params': params,
    'train': (xy_w, bnums),
    'test': (xy_w_test, bnums_test)
}

outname = 'models/sls_model_1.pickle'
# outname = 'models/byfly2/{}.pickle'.format(fly)
# with open(outname, 'wb') as f:
    # pickle.dump(model_standard, f)


n_ang = len(angles_main)

common = Counter(bnums).most_common(10)
b = common[0][0]

n_pred = 100

cc = np.where(b == bnums)[0][:n_pred]
real_ang = xy_w[0][cc, :n_ang]
real_drv = xy_w[0][cc, n_ang:n_ang*2]
rcos, rsin = xy_w[0][:, [-2, -1]][cc].T
real_phase = np.arctan2(rsin, rcos)
real_context = xy_w[0][cc, -3:-2]

ang = real_ang[0]
drv = real_drv[0]
context = real_context
pcos, psin = rcos[0], rsin[0]
phase = np.arctan2(psin, pcos)

pred_ang = np.zeros((n_pred, ang.shape[-1]))
pred_drv = np.zeros((n_pred, drv.shape[-1]))
pred_phase = np.zeros(n_pred)

for i in range(n_pred):
  out = model_walk(np.hstack([ang, drv, context[i], pcos, psin])[None])[0].numpy()
  ang = out[:n_ang]
  drv = out[n_ang:n_ang*2]
  phase = np.mod(phase + out[-1], 2*np.pi)
  # phase = real_phase[i]
  pcos, psin = np.cos(phase), np.sin(phase)
  pred_ang[i] = ang
  pred_drv[i] = drv
  pred_phase[i] = phase


import matplotlib.pyplot as plt
plt.figure(1)
plt.clf()
plt.subplot(211)
plt.plot(np.cos(pred_phase))
plt.plot(np.sin(pred_phase))
plt.subplot(212)
plt.plot(np.cos(real_phase))
plt.plot(np.sin(real_phase))
plt.draw()
plt.show(block=False)

plt.figure(1)
plt.clf()
plt.subplot(211)
plt.plot(pred_ang)
plt.subplot(212)
plt.plot(real_ang)
plt.draw()
plt.show(block=False)
