#!/usr/bin/env python3

import pandas as pd
import numpy as np

from tqdm import tqdm, trange
from scipy import stats, signal
import os
from collections import Counter, defaultdict
import pickle
import time
import sys

import argparse
import sys
project_path = "/home/lili/research/tuthill/layered-walking"
sys.path.append(project_path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.object = object
np.bool = bool
np.int = int

import tensorflow as tf

# supress warning messages
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from tools.model_functions import get_props_by_bouts, get_sw_xy, get_period, get_speed
from tools.model_functions import prop_params
from tools.model_functions import MLPScaledXY, num_vars, PFMLPScaledXY
from tools.model_functions import wrap_array, summarize
from tools.trajgen_tools import WalkingData

# print("TensorFlow version: {}".format(tf.__version__))

# i need these lines so that the GPU doesn't take up the whole memory in tensorflow
gpus = tf.config.list_physical_devices('GPU')
# print(gpus)
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



data_filename = '/home/lili/data/tuthill/models/models_sls/data_subang_5.pickle'

outdir = '/home/lili/data/tuthill/models/models_sls_sweep_v3'
os.makedirs(outdir, exist_ok=True)


legs = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']
all_angles_main = [
    ['L1A_abduct', 'L1A_rot', 'L1B_flex', 'L1C_flex'],
    ['L2B_flex', 'L2B_rot', 'L2C_flex'],
    ['L3B_flex', 'L3B_rot', 'L3C_flex'],
    ['R1A_abduct', 'R1A_rot', 'R1B_flex', 'R1C_flex'],
    ['R2B_flex', 'R2B_rot', 'R2C_flex'],
    ['R3B_flex', 'R3B_rot', 'R3C_flex']
]

# key parameters

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension size')
parser.add_argument('--dropout_rate', type=float, default=0.05, help='Dropout rate')
parser.add_argument('--phase_noise', type=float, default=0.25, help='Phase noise value')
parser.add_argument('--n_pred_btt', type=int, default=0, help='Number of backprop through time iterations')
parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs for training')

args, unknown = parser.parse_known_args()

hidden_dim = args.hidden_dim
dropout_rate = args.dropout_rate
phase_noise = args.phase_noise
n_pred_btt = args.n_pred_btt
n_epochs = args.n_epochs

train_params = {
    'hidden_dim': hidden_dim,
    'dropout_rate': dropout_rate,
    'phase_noise': phase_noise,
    'n_pred_btt': n_pred_btt,
    'n_epochs': n_epochs
}

param_names = [
    ('hidden_dim', 'hd', 1),
    ('dropout_rate', 'dr', 100),
    ('phase_noise', 'pn', 100),
    ('n_pred_btt', 'np', 1),
    ('n_epochs', 'ep', 1)
]

def make_outname(params):
    outname = 'model'
    for name, short, mult in param_names:
        val = '_{}{:03d}'.format(short, int(params[name] * mult))
        outname += val
    outname += '.pkl'
    return outname

outname = make_outname(train_params)
outpath = os.path.join(outdir, outname)


print()
print(train_params)
print(outpath)

if os.path.exists(outpath):
  print('already exists, exiting!')
  exit()

wd = WalkingData(data_filename)

xy_ws = []
for leg in legs:
    xy_w, bnums = wd.data[leg]['train']
    xy_ws.append(xy_w)

ms_walk = []
for xy_w in xy_ws:
    model_walk = MLPScaledXY(output_dim=xy_w[1].shape[1],
                             hidden_dim=hidden_dim, dropout_rate=dropout_rate,
                             msx=xy_w[2], msy=xy_w[3])
    ms_walk.append(model_walk)
    model_walk(xy_w[0][:2])
    print('Walk', num_vars(model_walk))


## single time step training
def make_step():
    lr = tf.Variable(1e-3)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    @tf.function
    def step_mlp_norm(model_walk, in_walk, out_walk):
      """Performs one optimizer step on a single mini-batch."""
      with tf.GradientTape() as tape:
          pred_walk = model_walk(in_walk, is_training=True)
          error_walk = tf.square(out_walk - pred_walk) / tf.square(model_walk.msy[1])
          loss = tf.reduce_mean(error_walk)

      variables = model_walk.trainable_variables
      grads = tape.gradient(loss, variables)
      opt.apply_gradients(zip(grads, variables))
      return loss
    return step_mlp_norm


batch_size = 10000

for leg, model_walk, xy_w in zip(legs, ms_walk, xy_ws):
    step_mlp_norm = make_step()
    print(leg)
    in_walk = xy_w[0]
    out_walk = xy_w[1]

    stds = np.std(in_walk, axis=0)

    t0 = time.time()

    for epoch_num in range(n_epochs+1):
        ixs = np.arange(len(in_walk))
        np.random.shuffle(ixs)
        total = 0
        num = 0
        for s in range(0, len(ixs), batch_size):
            c = ixs[s:s+batch_size]
            # in_walk_c = tf.cast(in_walk[c], 'float32')
            # out_walk_c = tf.cast(out_walk[c], 'float32')
            in_walk_c = in_walk[c].astype('float32')
            out_walk_c = out_walk[c].astype('float32')
            # add phase noise
            p = np.arctan2(in_walk_c[:, -1], in_walk_c[:, -2])
            p += np.random.normal(size=p.shape) * phase_noise
            in_walk_c[:, -1] = np.sin(p)
            in_walk_c[:, -2] = np.cos(p)
            total += step_mlp_norm(model_walk, in_walk_c, out_walk_c).numpy()
            num += 1
        if epoch_num % 50 == 0:
            t1 = time.time() - t0
            print("Time: {:.2f} Epoch {}: {:.5f}".format(t1, epoch_num, total / num))

    print("Done!")



## backprop through time

def make_btt_step(n_ang):
    lr_fn = tf.Variable(1e-3)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_fn, clipnorm=10.0)

    @tf.function
    @tf.autograph.experimental.do_not_convert
    def step_model(model_walk, in_walks):
        scales = model_walk.msx[1]
        scale_ang = np.mean(scales[:n_ang*2])
        scale_drv = np.mean(scales[n_ang*2:n_ang*3])
        scale_phase = np.mean(scales[-2:])

        real_ang_c = in_walks[:, :, :n_ang]
        real_ang_s = in_walks[:, :, n_ang:n_ang*2]

        real_ang = tf.atan2(real_ang_s, real_ang_c) * 180 / np.pi

        real_drv = in_walks[:, :, n_ang*2:n_ang*3]
        real_cos = in_walks[:, :, -2]
        real_sin = in_walks[:, :, -1]
        context = in_walks[:, :, -5:-2]
        real_phase = tf.atan2(real_sin, real_cos)

        n_pred = np.shape(in_walks)[0]

        ang = real_ang[0]
        drv = real_drv[0]
        phase = real_phase[0][:,None]

        with tf.GradientTape() as tape:
            losses = defaultdict(float)
            for i in range(n_pred - 1):
                prev_phase = phase
                rad = ang * np.pi / 180
                inp = tf.concat([tf.cos(rad), tf.sin(rad), drv,
                                 context[i], tf.cos(phase), tf.sin(phase)], axis=1)
                out = model_walk(inp)
                ang1 = ang + drv * 0.5
                drv1 = drv + out[:,:n_ang] * 0.5
                phase1 = phase + out[:,-1:] * 0.5

                rad1 = ang1 * np.pi / 180
                inp = tf.concat([tf.cos(rad1), tf.sin(rad1), drv1,
                                 context[i], tf.cos(phase1), tf.sin(phase1)], axis=1)
                out = model_walk(inp)
                ang = ang + drv * 1.0
                drv = drv + out[:,:n_ang] * 1.0
                phase = phase + out[:,-1:] * 1.0

                rad = ang * np.pi / 180

                losses['ang_c'] += tf.reduce_mean(tf.square(tf.cos(rad) - real_ang_c[i+1])) / tf.square(scale_ang)
                losses['ang_s'] += tf.reduce_mean(tf.square(tf.sin(rad) - real_ang_s[i+1])) / tf.square(scale_ang)
                losses['drv'] += tf.reduce_mean(tf.square(drv - real_drv[i+1])) / tf.square(scale_drv)
                losses['phase_c'] += tf.reduce_mean(tf.abs(tf.cos(phase) - tf.cos(real_phase[i+1]))) / scale_phase
                losses['phase_s'] += tf.reduce_mean(tf.abs(tf.sin(phase) - tf.sin(real_phase[i+1]))) / scale_phase
                # losses['phase_cd'] += tf.reduce_mean(tf.abs(tf.cos(phase - prev_phase) -
                #                                            tf.cos(real_phase[i+1] - real_phase[i]))) / scale_phase * 10
                # losses['phase_sd'] += tf.reduce_mean(tf.square(
                #     tf.sin(phase - prev_phase) -
                #     tf.sin(real_phase[i+1] - real_phase[i]))) / scale_phase * 10
                # if toadd < 10:
                #     loss += toadd
                # else:
                #     loss += 100.0

            loss = 0.0
            for k, v in losses.items():
                if k in ['ang_c', 'ang_s']:
                    loss += v
            variables = model_walk.trainable_variables
            # loss += tf.reduce_mean([ tf.nn.l2_loss(v) for v in variables
            #                         if 'bias' not in v.name ]) * 0.01

        # if tf.math.is_nan(loss):
          # return losses


        grads = tape.gradient(loss, variables)
        tf.cond(tf.math.is_nan(loss),
                lambda: tf.no_op(),
                lambda: opt.apply_gradients(zip(grads, variables)))

        # grads = tape.gradient(loss, variables)
        # opt.apply_gradients(zip(grads, variables))
        return losses

    return step_model

n_batch_btt = 2500
# n_epochs_btt = 2501
n_epochs_btt = 1001
interval_btt = 250

if n_pred_btt > 0:
    all_in_walks = []
    for xy_w in tqdm(xy_ws, ncols=70):
        in_walks = []
        for b in np.unique(bnums):
            cc = np.where(b == bnums)[0]
            in_walk = xy_w[0][cc]
            for i in range(0, len(in_walk)-n_pred_btt, 20):
                in_walks.append(in_walk[i:n_pred_btt+i])
        in_walks = np.stack(in_walks, axis=1)
        all_in_walks.append(in_walks)


    for leg, model_walk, in_walks, angles_main in zip(legs, ms_walk, all_in_walks, all_angles_main):
        step_model = make_btt_step(len(angles_main))
        print(leg)
        totals = defaultdict(float)
        count = 0
        for enum in range(n_epochs_btt):
            ixs = np.random.choice(in_walks.shape[1], size=n_batch_btt, replace=False)

            losses = step_model(model_walk, in_walks[:, ixs])
            real_ang = in_walks[:, ixs, 0]
            # pred_ang = np.vstack(pred_ang)
            for k, v in losses.items():
                totals[k] += v.numpy()
            # total += loss.numpy()
            count += 1
            if enum % interval_btt == 0:
                # means = dict()
                s = ""
                for k, v in totals.items():
                    s = s+ "  {}: {:.3f}".format(k, v/count)
                    # means[k] = v / count
                    totals[k] = 0
                # print(enum, total.mean() / count)
                print("{: 5d}".format(enum), s)
                total = 0
                count = 0




all_models = dict()
all_models['train_params'] = train_params
for leg, model_walk, angles_main in zip(legs, ms_walk, all_angles_main):
    all_models[leg] = {
        'model_walk': model_walk.get_full(),
        'angle_names': angles_main
    }

with open(outpath, 'wb') as f:
    pickle.dump(all_models, f)
