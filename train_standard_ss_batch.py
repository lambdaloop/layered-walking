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
from model_functions import prop_params
from model_functions import MLPScaledXY, num_vars

# fly = '6.15.20 Fly 4_0'
# fly = sys.argv[1]

# prob_perturb = 0.5
# accel_ratio = 1.0
# sd_error_deriv_scale = 0.5
# n_epochs = 200

prob_perturb = float(sys.argv[1])
accel_ratio = float(sys.argv[2])
sd_error_deriv_scale = float(sys.argv[3])

n_epochs = int(sys.argv[4])

if len(sys.argv) >= 6:
  num_add = int(sys.argv[5])
else:
  num_add = -1
#

if len(sys.argv) >= 7:
  use_state = int(sys.argv[6]) != 0
else:
  use_state = True

if len(sys.argv) >= 8:
  use_phase = int(sys.argv[7]) != 0
else:
  use_phase = True

# print("TensorFlow version: {}".format(tf.__version__))

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
with open('models-full/full_model_corrected_extreme.pickle', 'rb') as f:
# with open('models/byfly2/6.15.20 Fly 4_0.pickle', 'rb') as f:
    model_ref = pickle.load(f)


FPS = 300.0

(xy_s, xy_w), bnums = model_ref['train']
(xy_s_test, xy_w_test), bnums_test = model_ref['test']

# params = {'context': ['fictrac_speed', 'fictrac_rot'], 'use_phase': True}
# params = {'context': ['fictrac_speed'], 'use_phase': True}
params = {'context': ['fictrac_speed'], 'use_phase': use_phase, 'use_state': use_state}

# (xy_s, xy_w), bnums = get_data(good_bouts[:-5], params)
# (xy_s_test, xy_w_test), bnums_test = get_data(good_bouts[-5:], params)

# print("Data points in training set:", len(xy_s[0]))
# print("Data points in test set:", len(xy_s_test[0]))
if not params['use_phase']: # remove phase signals
  xy_w = list(xy_w)
  xy_w_test = list(xy_w_test)
  xy_w[0] = xy_w[0][:, :-2]
  xy_w_test[0] = xy_w_test[0][:, :-2]
  xy_w[1] = xy_w[1][:, :1]
  xy_w_test[1] = xy_w_test[1][:, :1]
  xy_w[2] = (xy_w[2][0][:-2], xy_w[2][1][:-2])
  xy_w_test[2] = (xy_w_test[2][0][:-2], xy_w_test[2][1][:-2])
  xy_w[3] = (xy_w[3][0][:1], xy_w[3][1][:1])
  xy_w_test[3] = (xy_w_test[3][0][:1], xy_w_test[3][1][:1])

if not params['use_state']:
  xy_s = list(xy_s)
  xy_s_test = list(xy_s_test)
  subix = [0, 1, 2, 3, 4, 7]
  xy_s[0] = xy_s[0][:, subix]
  xy_s_test[0] = xy_s_test[0][:, subix]
  xy_s[2] = (xy_s[2][0][subix], xy_s[2][1][subix])
  xy_s_test[2] = (xy_s_test[2][0][subix], xy_s_test[2][1][subix])



model_state = MLPScaledXY(output_dim=2, hidden_dim=256, dropout_rate=0.05,
                          msx=xy_s[2], msy=xy_s[3])

model_walk = MLPScaledXY(output_dim=1 + params['use_phase'], hidden_dim=256, dropout_rate=0.05,
                         msx=xy_w[2], msy=xy_w[3])


model_state(xy_s[0][:2])
# print('State', num_vars(model_state))

model_walk(xy_w[0][:2])
# print('Walk', num_vars(model_walk))


in_state = xy_s[0]
out_state = xy_s[1]
in_walk = xy_w[0]
extra_walk = in_walk[:, 2:]
out_walk = xy_w[1]



# prob_perturb = 0.5
# accel_ratio = 1.5
# sd_err_deriv_scale = 0.5

sd_err_deriv = np.std(out_state[:,1]) * sd_error_deriv_scale

lr = tf.Variable(1e-3)
opt = tf.keras.optimizers.Adam(learning_rate=lr)

@tf.function
def step_mlp_norm(model_state, model_walk,
                  in_state, extra_walk,
                  out_state, out_walk,
                  toadd_state, toadd_walk):
  """Performs one optimizer step on a single mini-batch."""
  toadd_state = tf.cast(toadd_state, 'float32')
  toadd_walk = tf.cast(toadd_walk, 'float32')

  in_state = tf.cast(in_state, 'float32')
  extra_walk = tf.cast(extra_walk, 'float32')
  out_state = tf.cast(out_state, 'float32')
  out_walk = tf.cast(out_walk, 'float32')


  with tf.GradientTape() as tape:
      pred_state = model_state(in_state, is_training=True)
      error_state = tf.square(out_state - pred_state) / model_state.msy[1]

      in_walk_1 = tf.concat([pred_state, extra_walk], axis=1)
      pred_walk_1 = model_walk(in_walk_1, is_training=True)
      error_walk_1 = tf.square(out_walk - pred_walk_1) / model_walk.msy[1]

      in_walk_2 = tf.concat([out_state + toadd_state, extra_walk], axis=1)
      pred_walk_2 = model_walk(in_walk_2, is_training=True)
      error_walk_2 = tf.square((out_walk+toadd_walk) - pred_walk_2) / model_walk.msy[1]

      loss = tf.reduce_mean(error_state)
      loss += 0.5 * tf.reduce_mean(error_walk_1)
      loss += 0.5 * tf.reduce_mean(error_walk_2)

  variables = model_state.trainable_variables + model_walk.trainable_variables
  grads = tape.gradient(loss, variables)
  opt.apply_gradients(zip(grads, variables))
  return loss


# batch_size = 2500
# n_epochs = 6000
batch_size = 8000
# n_epochs = 200
# n_epochs = 1000


t0 = time.time()

for epoch_num in range(n_epochs+1):
    ixs = np.arange(len(in_state))
    np.random.shuffle(ixs)
    total = 0
    num = 0
    for s in range(0, len(ixs), batch_size):
        c = ixs[s:s+batch_size]
        in_state_c = np.copy(in_state[c])
        extra_walk_c = np.copy(extra_walk[c])
        out_state_c = np.copy(out_state[c])
        out_walk_c = np.copy(out_walk[c])

        N = len(c)
        perturb = np.random.uniform(size=N) < prob_perturb
        toadd_state = np.zeros(out_state_c.shape)
        toadd_walk = np.zeros(out_walk_c.shape)
        toadd_state[:, 1] += perturb * np.random.normal(size=N) * sd_err_deriv
        toadd_walk[:, 0] -= toadd_state[:, 1] * accel_ratio

        total += step_mlp_norm(model_state, model_walk,
                               in_state_c, extra_walk_c,
                               out_state_c, out_walk_c,
                               toadd_state, toadd_walk).numpy()

        num += 1
    if epoch_num % 25 == 0:
        t1 = time.time() - t0
        print("Time: {:.2f} Epoch {}: {:.5f}".format(t1, epoch_num, total / num))


model_standard = {
    # 'fly': fly,
    'model_walk': model_walk.get_full(),
    'model_state': model_state.get_full(),
    # 'prop_params': (claw_flex_basic, claw_ext_basic, club_squash, hook_flex_squash),
    'prop_params': prop_params,
    'params': params,
    'train': ((xy_s, xy_w), bnums),
    'test': ((xy_s_test, xy_w_test), bnums_test)
}

# outname = 'models/full_model_corrected_extreme_2.pickle'
# outname = 'models/byfly2/{}.pickle'.format(fly)

if num_add < 0:
  outname = 'models_correction/model_p{}_a{}_e{}_n{}_sp{:d}{:d}.pickle'.format(
    int(prob_perturb*100), int(accel_ratio*10), int(sd_error_deriv_scale*100), n_epochs,
    use_state, use_phase)
else:
  outname = 'models_ensemble/model_p{}_a{}_e{}_n{}_sp{:d}{:d}_i{}.pickle'.format(
    int(prob_perturb*100), int(accel_ratio*10), int(sd_error_deriv_scale*100),
    n_epochs, use_state, use_phase, num_add)
with open(outname, 'wb') as f:
    pickle.dump(model_standard, f)
