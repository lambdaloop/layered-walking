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
import pandas as pd

# function:
# - takes in model
# - phase angle of model in standard walking
# - evaluate KS of model in a few standard walking assays
# - evaluates KS of model under a few perturbations
# - plot a few videos to see the model

## setup
data_filename = '/home/lili/data/tuthill/models/models_sls/data_subang_1.pickle'

tg_filename = '/home/lili/data/tuthill/models/models_sls/walk_sls_legs_subang_10.pickle'

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

def simulate_bout(TG, bout, n_pred=500, dists=[], coupling_ratio=1):
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
        px_half = px + 0.5*Ts * kuramato_deriv(px, alphas, offsets, ws) * coupling_ratio
        toadd = Ts * kuramato_deriv(px_half, alphas, offsets, ws) * coupling_ratio
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


## evaluation

## phase angle evaluation
xvals = np.linspace(-np.pi, np.pi)
def get_phase(ang):
    m = np.median(ang, axis=0)
    s = np.std(ang, axis=0)
    s[s == 0] = 1
    dm = (ang - m) / s
    phase = np.arctan2(-dm[:,1], dm[:,0])
    return phase

def phase_align_poly(ang, extra=None, order=5):
    topredict = ang
    if extra is not None:
        topredict = np.hstack([ang, extra])
    means = np.full((len(xvals), topredict.shape[1]), np.nan)
    stds = np.full((len(xvals), topredict.shape[1]), np.nan)
    if len(ang) < 50: # not enough data
        return means, stds
    phase = get_phase(ang)
    # pcs = np.vstack([np.cos(phase), np.sin(phase)]).T
    b = np.vstack([np.cos(i * phase + j) for i in range(order) for j in [0, np.pi/2]]).T
    xcs = np.vstack([np.cos(i * xvals + j) for i in range(order) for j in [0, np.pi/2]]).T
    for i in range(topredict.shape[1]):
        cc = np.isfinite(topredict[:,i])
        model = sm.OLS(topredict[cc,i], b[cc]).fit()
        means[:,i] = model.predict(xcs)
        s, upper,lower = wls_prediction_std(model, xcs)
        stds[:,i] = s
    return means, stds

def circular_mean(x):
    # return np.degrees(stats.circmean(np.radians(x), nan_policy='omit'))
    return np.degrees(np.angle(np.nanmean(np.exp(1j * np.radians(x)))))


def get_phase_means(bout):
    n_legs = len(legs)
    dof   = 5
    means  = np.full((n_legs, dof, len(xvals)), np.nan)
    drvs  = np.full((n_legs, dof, len(xvals)), np.nan)

    for ix_leg, leg in enumerate(legs):
        numAng = TG[ix_leg]._numAng
        for ix_ang in range(numAng):
            ang = bout['angles'][legs[ix_leg]][:n_pred, ix_ang]
            drv = bout['derivatives'][legs[ix_leg]][:n_pred, ix_ang]
            x = np.vstack([ang, drv]).T
            m, _ = phase_align_poly(x)
            means[ix_leg, ix_ang] = m[:, 0]
            drvs[ix_leg, ix_ang] = m[:, 1]

    return means, drvs


def ks_bout(bout):
    angs = np.hstack([bout['angles'][leg] for leg in legs])
    angs_sc = np.hstack([np.sin(np.deg2rad(angs)),
                      np.cos(np.deg2rad(angs))])
    pangs = wd.data['pca'].transform(angs_sc)
    pdfs = wd.data['kde'].logpdf(pangs.T)
    return pdfs

# pretty high perturbations, to really evaluate model
# it's decently robust anyway in the case of no delays here
distDict = {
    'maxVelocity' : 10,
    'rate': 20 * Ts,
    'start': 0,
    'end': n_pred
}

video_speeds = [
    [8, 0, 0], [10, 0, 0], [12, 0, 0], [14, 0, 0],
    # [12, -8, 0], [12, 8, 0], [12, 0, -4], [12, 0, 4]
]

all_params = [
    {'speed_x': speed[0],
     'speed_y': speed[1],
     'speed_z': speed[2],
     'offset': offset,
     'dist': dist_enabled,
     'coupling_ratio': coupling_ratio}

    for dist_enabled in [True, False]
    for coupling_ratio in [0, 1]
    for speed in video_speeds
    for offset in range(2)
]

all_errors = []

for params in tqdm(all_params, ncols=70):
    # speed = video_speeds[0]
    # offset = 3
    row = dict(params)

    if row['dist']:
        dists = [distDict]
    else:
        dists = []

    speed = [row['speed_x'], row['speed_y'], row['speed_z']]

    bout_real = wd.get_bout(speed, offset=row['offset'], min_bout_length=n_pred)
    bout_sim = simulate_bout(TG, bout_real, dists=dists, coupling_ratio=row['coupling_ratio'])

    means_sim, drvs_sim  = get_phase_means(bout_sim)
    means_real, drvs_real  = get_phase_means(bout_real)

    # compute error
    errors = np.mean(np.abs(means_sim - means_real), axis=2).ravel()
    row['angle_error'] = np.nanmean(errors)

    errors = np.mean(np.abs(drvs_sim - drvs_real), axis=2).ravel()
    row['deriv_error'] = np.nanmean(errors) * 300

    ## KS evaluation
    pdfs_sim = ks_bout(bout_sim)
    pdfs_real = ks_bout(bout_real)[:n_pred]
    row['ks_sim'] = np.mean(pdfs_sim)
    row['ks_real'] = np.mean(pdfs_real)

    all_errors.append(row)
