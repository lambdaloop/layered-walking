#!/usr/bin/env python3


# only 1 thread, to help parallelize across data
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress warnings from tensorflow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # compute on cpu, it's actually faster for inference with smaller model


from tools.trajgen_tools import TrajectoryGenerator, WalkingData
from tools.angle_functions import legs, \
                            offsets, alphas, kuramato_deriv, \
                            angles_to_pose_names, make_fly_video, \
                            ctrl_to_tg
import tensorflow as tf

import pickle
import numpy as np
from tqdm import trange, tqdm

import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# function:
# - takes in model
# - phase angle of model in standard walking
# - evaluate KS of model in a few standard walking assays
# - evaluates KS of model under a few perturbations
# - plot a few videos to see the model

## setup
data_filename = '/home/lili/data/tuthill/models/models_sls/data_subang_2.pickle'
# tg_filename = '/home/lili/data/tuthill/models/models_sls/walk_sls_legs_subang_6.pickle'
# tg_filename = '/home/lili/data/tuthill/models/models_sls_sweep/model_hd016_dr010_pn070_np000_ep100.pkl'
# tg_filename = '/home/lili/data/tuthill/models/models_sls_sweep/model_hd016_dr000_pn090_np000_ep100.pkl'
# tg_filename = '/home/lili/data/tuthill/models/models_sls_sweep/model_hd016_dr000_pn040_np000.pkl'
# tg_filename = '/home/lili/data/tuthill/models/models_sls_sweep_moreiters/model_hd256_dr005_pn020_np000_ep1000.pkl'
# tg_filename = '/home/lili/data/tuthill/models/models_sls_sweep/model_hd064_dr000_pn050_np000_ep100.pkl'
# tg_filename = '/home/lili/data/tuthill/models/models_sls_sweep/model_hd016_dr000_pn030_np000.pkl'
tg_filename = '/home/lili/data/tuthill/models/models_sls_sweep/model_hd032_dr000_pn030_np000.pkl'
# tg_filename = '/home/lili/data/tuthill/models/models_sls_sweep_moreiters/model_hd128_dr000_pn020_np000_ep1000.pkl'


legs = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']

n_pred = 500

wd = WalkingData(data_filename)
TG = [TrajectoryGenerator(tg_filename, leg, n_pred) for leg in legs]

Ts = 1/300 # How fast TG runs


## running

def get_dist(d, numAng):
    occur = np.random.poisson(d['rate'])
    if occur == 0:
        return np.zeros(numAng)
    dists = np.random.normal(d['maxVelocity'], d['maxVelocity']/10, numAng)
    sign = np.sign(np.random.uniform(-1, 1, size=numAng))
    dists = dists * sign
    return dists

def simulate_bout(TG, bout, n_pred=500, dists=[]):
    n_legs   = len(legs)
    dof   = 5

    contexts = bout['contexts'].astype('float32')

    angleTG = np.zeros((n_legs, dof, n_pred), dtype='float32')
    drvTG   = np.zeros((n_legs, dof, n_pred), dtype='float32')
    phaseTG = np.zeros((n_legs, n_pred), dtype='float32')

    for ln, leg in enumerate(legs):
        numAng = TG[ln]._numAng
        angleTG[ln,:numAng,0] = bout['angles'][leg][0]
        drvTG[ln,:numAng,0] = bout['derivatives'][leg][0]
        phaseTG[ln,0] = bout['phases'][leg][0]


    for k in range(n_pred-1):
        # kuramoto sync
        ws = np.zeros(6)
        px = phaseTG[:,k]
        px_half = px + 0.5*Ts * kuramato_deriv(px, alphas, offsets, ws)
        toadd = Ts * kuramato_deriv(px_half, alphas, offsets, ws) 
        px = px + toadd
        phaseTG[:,k] = px

        for ln, (leg, model) in enumerate(zip(legs, TG)):
            numAng = TG[ln]._numAng
            ang = angleTG[ln,:numAng,k]
            drv = drvTG[ln,:numAng,k]
            phase = phaseTG[ln, k]
            context = contexts[k]

            ang_new, drv_new, phase_new = TG[ln].step_forward(ang, drv, phase, context)
            angleTG[ln, :numAng, k+1] = ang_new
            drvTG[ln, :numAng, k+1] = drv_new
            phaseTG[ln, k+1] = phase_new

            for dist in dists:
                if k >= dist['start'] and k < dist['end']:
                    add = get_dist(dist, numAng)
                    angleTG[ln, :numAng, k+1] += add

    bout_sim = dict()
    bout_sim['contexts'] = bout['contexts']
    bout_sim['angles'] = dict()
    bout_sim['derivatives'] = dict()
    bout_sim['phases'] = dict()

    for ln, leg in enumerate(legs):
        numAng = TG[ln]._numAng
        bout_sim['angles'][leg] = angleTG[ln, :numAng].T
        bout_sim['derivatives'][leg] = drvTG[ln, :numAng].T
        bout_sim['phases'][leg] = phaseTG[ln]

    return bout_sim

# pretty high perturbations, to really evaluate model
# it's decently robust anyway in the case of no delays here
distDict = {
    'maxVelocity' : 10,
    'rate': 20 * Ts,
    'start': 0,
    'end': n_pred
}

speeds = [10, 12, 14]
bouts_real = []
bouts_sim = []

for speed in speeds:
    speed_list = [speed, 0, 0]
    bout = wd.get_bout(speed_list, offset=0, min_bout_length=n_pred)
    bout_sim = simulate_bout(TG, bout, dists=[])
    bouts_real.append(bout)
    bouts_sim.append(bout_sim)

## evaluation

## qualitative plot evaluation
ix_leg = 0
ix_ang = 3

print(wd.data[legs[ix_leg]]['angle_names'][ix_ang])

plt.figure(1)
plt.clf()
for i, (speed, bout, bout_sim) in enumerate(zip(speeds, bouts_real, bouts_sim)):
    plt.subplot(len(speeds), 1, i+1)
    ang_sim = bout_sim['angles'][legs[ix_leg]][:n_pred, ix_ang]
    # ang_sim = np.mod(ang_sim, 360)
    plt.plot(ang_sim, label='simulated')
    plt.plot(bout['angles'][legs[ix_leg]][:n_pred, ix_ang], label='real')
    plt.title("{} mm/s".format(speed))
plt.legend()
plt.ylabel('femur-tibia flexian (deg)')
plt.xlabel('frames')
plt.draw()
plt.tight_layout()
plt.show(block=False)


