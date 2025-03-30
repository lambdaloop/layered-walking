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
from scipy import signal

# function:
# - takes in model
# - phase angle of model in standard walking
# - evaluate KS of model in a few standard walking assays
# - evaluates KS of model under a few perturbations
# - plot a few videos to see the model

## setup
data_filename = '/home/lili/data/tuthill/models/models_sls/data_subang_2.pickle'
# tg_filename = '/home/lili/data/tuthill/models/models_sls/walk_sls_legs_subang_6.pickle'
# tg_filename = '/home/lili/data/tuthill/models/models_sls_sweep/model_hd016_dr000_pn000_np000.pkl'
tg_filename = '/home/lili/data/tuthill/models/models_sls_sweep/model_hd032_dr000_pn030_np000.pkl'

legs = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']

n_pred = 500

wd = WalkingData(data_filename)
TG = [TrajectoryGenerator(tg_filename, leg, n_pred) for leg in legs]

Ts = 1/300 # How fast TG runs

speed = [19, 0, 0]
bout = wd.get_bout(speed, offset=0, min_bout_length=n_pred)


# pretty high perturbations, to really evaluate model
# it's decently robust anyway in the case of no delays here
distDict = {
    'maxVelocity' : 10,
    'rate': 20 * Ts,
    'start': 0,
    'end': n_pred
}

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

bout_sim = simulate_bout(TG, bout, dists=[])

## evaluation

## qualitative plot evaluation
ix_leg = 0
ix_ang = 3

print(wd.data[legs[ix_leg]]['angle_names'][ix_ang])

ang_sim = bout_sim['angles'][legs[ix_leg]][:n_pred, ix_ang]
ang_real = bout['angles'][legs[ix_leg]][:n_pred, ix_ang]

plt.figure(1)
plt.clf()
plt.plot(bout_sim['angles'][legs[ix_leg]][:n_pred, ix_ang])
plt.plot(bout['angles'][legs[ix_leg]][:n_pred, ix_ang])
plt.ylabel('femur-tibia flexian (deg)')
plt.xlabel('frames')
plt.draw()
plt.show(block=False)

pose_real = bout_to_pose(bout)
pose_sim = bout_to_pose(bout_sim)

def rough_range(ang):
    low, high = np.percentile(ang, [2, 98])
    return high - low

def step_length(bout):
    pose = bout_to_pose(bout)
    y = pose[:, 0, 4, 1] # L1 tip y
    return rough_range(y)

def step_frequency(bout):
    ang = bout['angles']['L1'][:, 3] # L1 femur-tibia flexion
    peaks, heights = signal.find_peaks(ang, 70)
    period = np.mean(np.diff(peaks)) * Ts
    freq = 1 / period
    return freq

plt.figure(1)
plt.clf()
plt.plot(pose_sim[:, 0, 4, 1])
plt.plot(pose_real[:, 0, 4, 1])
plt.draw()
plt.show(block=False)



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

all_angles_main = [wd.data[leg]['angle_names'] for leg in legs]

def bout_to_pose(bout):
    angles = np.hstack([bout['angles'][leg] for leg in legs])
    angle_names = np.hstack(all_angles_main)
    pose = angles_to_pose_names(angles, angle_names)
    return pose


n_legs = len(legs)
dof   = 5
means_sim  = np.zeros((n_legs, dof, len(xvals)))
means_real = np.zeros((n_legs, dof, len(xvals)))

drvs_sim  = np.zeros((n_legs, dof, len(xvals)))
drvs_real = np.zeros((n_legs, dof, len(xvals)))

for ix_leg, leg in enumerate(legs):
    numAng = TG[ix_leg]._numAng
    for ix_ang in range(numAng):
        # get sim
        ang = bout_sim['angles'][legs[ix_leg]][:n_pred, ix_ang]
        drv = bout_sim['derivatives'][legs[ix_leg]][:n_pred, ix_ang]
        x = np.vstack([ang, drv]).T
        m, _ = phase_align_poly(x)
        means_sim[ix_leg, ix_ang] = m[:, 0]
        drvs_sim[ix_leg, ix_ang] = m[:, 1]

        # get real
        ang = bout['angles'][legs[ix_leg]][:n_pred, ix_ang]
        drv = bout['derivatives'][legs[ix_leg]][:n_pred, ix_ang]
        x = np.vstack([ang, drv]).T
        m, _ = phase_align_poly(x)
        means_real[ix_leg, ix_ang] = m[:, 0]
        drvs_real[ix_leg, ix_ang] = m[:, 1]

# compute error
errors = np.mean(np.abs(means_sim - means_real), axis=2).ravel()
valid = np.nonzero(errors)[0]
mean_err = np.mean(errors[valid])
print("Mean angle error: {:.2f} deg".format(mean_err))

errors = np.mean(np.abs(drvs_sim - drvs_real), axis=2).ravel()
valid = np.nonzero(errors)[0]
mean_err = np.mean(errors[valid])
print("Mean derivative error: {:.1f} deg/s".format(mean_err * 300))

# # plot
# fig, subplots = plt.subplots(n_legs, dof)
# for ix_leg, leg in enumerate(legs):
#     numAng = TG[ix_leg]._numAng
#     for ix_ang in range(numAng):
#         ax = subplots[ix_leg][ix_ang]
#         ax.plot(means_sim[ix_leg, ix_ang])
#         ax.plot(means_real[ix_leg, ix_ang])
#         name = wd.data[legs[ix_leg]]['angle_names'][ix_ang]
#         ax.set_title(name)
# plt.tight_layout()
# plt.draw()
# plt.show(block=False)

## KS evaluation

def ks_bout(bout):
    angs = np.hstack([bout['angles'][leg] for leg in legs])
    angs_sc = np.hstack([np.sin(np.deg2rad(angs)),
                      np.cos(np.deg2rad(angs))])
    pangs = wd.data['pca'].transform(angs_sc)
    pdfs = wd.data['kde'].logpdf(pangs.T)
    return pdfs

pdfs_sim = ks_bout(bout_sim)
pdfs_real = ks_bout(bout)[:n_pred]

print("Simulated angle KS : {:.2f}".format(np.mean(pdfs_sim)))
print("Real angle KS      : {:.2f}".format(np.mean(pdfs_real)))

## plot 
# plt.figure(1)
# plt.clf()
# plt.plot(pdfs_sim)
# plt.plot(pdfs_real)
# plt.draw()
# plt.show(block=False)

# plt.figure(2)
# plt.clf()
# _ = plt.hist(pdfs_sim, bins=50, density=True, histtype='step')
# _ = plt.hist(pdfs_real, bins=50, density=True, histtype='step')
# plt.draw()
# plt.show(block=False)

