#!/usr/bin/env python

import control
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag
from angle_functions import *
from ctrl_tools import *

# Sampling time
Ts = 1/300 # 300 times per second

# Assumes we already saved the linearized system in the appropriate files
# by running getLinearizedSystem()
ALin, BLin, xEqm, uEqm = loadLinearizedSystem()

dof = int(len(ALin)/2)
Nx  = 2*dof
Nu  = dof

# Discretize using e^(A*T) ~= I + A*T
A = np.eye(Nx) + ALin*Ts
B = Ts*BLin + 0.5*Ts*Ts*ALin @ BLin

eigsOL    = np.linalg.eig(A)[0]
specRadOL = max(np.abs(eigsOL))

# Sanity check: controllability
Qc = control.ctrb(A,B)
rankCtrb = np.linalg.matrix_rank(Qc)
if rankCtrb != Nx:
    print('Error: System uncontrollable!')

# Controller objective
anglePenalty    = 1e8
velocityPenalty = 1e-8
inputPenalty    = 1e-8

Q1 = anglePenalty*np.eye(dof)
Q2 = velocityPenalty*np.eye(dof)
Q  = block_diag(Q1, Q2)
R  = inputPenalty*np.eye(Nu)

K   = control.dlqr(A, B, Q, R)[0]
ACL = A - B @ K

# Sanity check: stability
eigsCL    = np.linalg.eig(ACL)[0]
specRadCL = max(np.abs(eigsCL))
if specRadCL >= 1:
    print('Error: Controller did not stabilize!')

n_pred = 500
time = np.array(range(n_pred))
ys   = np.zeros([Nx, n_pred])
us   = np.zeros([Nu, n_pred])

import pickle

# fn = '/home/lisa/Downloads/walk_sls_legs_2.pickle'
fn = '/home/pierre/data/tuthill/models/models_sls/walk_sls_legs_2.pickle'
with open(fn, 'rb') as f:
    allmodels = pickle.load(f)

# No warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ML Code
from model_functions import MLPScaledXY

angles_main = ['L1C_flex', 'L1A_rot', 'L1A_abduct', 'L1B_flex', 'L1B_rot']
angles_ctrl = ['L1A_abduct', 'L1B_flex', 'L1A_rot', 'L1C_flex']

def main2ctrl(angles):
    ctrlAngles = np.array([angles[2], angles[3], angles[1], angles[0]])
    return np.radians(ctrlAngles)

def ctrl2main(angles, L1BVal):
    # L1BVal is either angle or angular velocity for L1B_rot, which
    # is currently not in the model
        
    mainAngles = np.array([angles[3], angles[2], angles[0], angles[1]])
    return np.append(np.degrees(mainAngles), L1BVal)

model_walk = MLPScaledXY.from_full(allmodels['L1']['model_walk'])
xy_w, bnums = allmodels['L1']['train'] # xy_w is a real trajectory

from collections import Counter
n_ang = len(angles_main)
common = Counter(bnums).most_common(100)
b, _ = common[50]

cc = np.where(b == bnums)[0][:n_pred]

# Take code from ~line 528 in .org file (TG)
# ~line 800 contains CC+TG

# Generate trajectory using TG
real_ang = xy_w[0][cc, :n_ang]
real_drv = xy_w[0][cc, n_ang:n_ang*2]
rcos, rsin = xy_w[0][:, [-2, -1]][cc].T
real_phase = np.arctan2(rsin, rcos)
real_context = xy_w[0][cc, -4:-2]

ang = real_ang[0]
drv = real_drv[0]
context = real_context
pcos, psin = rcos[0], rsin[0]
phase = np.arctan2(psin, pcos)
phase_0 = phase

pred_ang = np.zeros((n_pred, n_ang))
pred_drv = np.zeros((n_pred, n_ang))
pred_phase = np.zeros(n_pred)

def update_state(ang, drv, phase, out, ratio=1.0):
    accel = out[:len(ang)]
    drv1 = drv + accel * ratio
    ang1 = ang + drv * ratio
    phase1 = phase + out[-1]*ratio
    return ang1, drv1, phase1

for i in range(n_pred):
    inp = np.hstack([ang, drv, context[i], np.cos(phase), np.sin(phase)])
    out = model_walk(inp[None].astype('float32'))[0].numpy()
    ang1, drv1, phase1 = update_state(ang, drv, phase, out, ratio=0.5)
    new_inp = np.hstack([ang1, drv1, context[i], np.cos(phase1), np.sin(phase1)])
    out = model_walk(new_inp[None].astype('float32'))[0].numpy()
    ang, drv, phase = update_state(ang, drv, phase, out, ratio=1.0)
    # phase = np.mod(real_phase[i], 2*np.pi)
    pred_ang[i] = ang
    pred_drv[i] = drv
    pred_phase[i] = phase

trajsTGOnly = np.zeros([Nx, n_pred])
for t in range(n_pred):
    trajsTGOnly[0:dof,t] = main2ctrl(pred_ang.T[:,t])
    trajsTGOnly[dof:,t]  = main2ctrl(pred_drv.T[:,t])

trajs = np.zeros([Nx, n_pred])
ang = real_ang[0]
drv = real_drv[0]
phase = phase_0

trajs[0:dof,0] = main2ctrl(ang)
trajs[dof:,0]  = main2ctrl(drv)

xEqmFlat = xEqm.flatten()
for t in range(n_pred-1):
    inp = np.hstack([ang, drv, context[t], np.cos(phase), np.sin(phase)])
    out = model_walk(inp[None].astype('float32'))[0].numpy()
    ang1, drv1, phase1 = update_state(ang, drv, phase, out, ratio=0.5)
    new_inp = np.hstack([ang1, drv1, context[t], np.cos(phase1), np.sin(phase1)])
    out = model_walk(new_inp[None].astype('float32'))[0].numpy()
    ang, drv, phase = update_state(ang, drv, phase, out, ratio=1.0)

    # phase = np.mod(real_phase[i], 2*np.pi)
    pred_ang[t] = ang
    pred_drv[t] = drv
    pred_phase[t] = phase

    trajs[0:dof,t+1] = main2ctrl(ang)
    trajs[dof:,t+1]  = main2ctrl(drv)
    
    wtraj        = A @ (trajs[:,t] - xEqmFlat) + xEqmFlat - trajs[:,t+1]
    
    # Give some look-ahead to wtraj
    us[:,t]      = -K @ (ys[:,t] + wtraj)    
    ys[:,t+1]    = A @ ys[:,t] + B @ us[:,t] + wtraj
    
    ang = ctrl2main(trajs[0:dof,t+1] + ys[0:dof,t+1], ang[-1])
    # drv = ctrl2main(trajs[dof:,t+1] + ys[dof:,t+1]*Ts, drv[-1])

qs = ys + trajs

angleErr    = np.linalg.norm(np.degrees(ys[0:dof,]), ord='fro')
velocityErr = np.linalg.norm(np.degrees(ys[dof:,]*Ts), ord='fro')

print(f'Frobenius norm of angle error: {angleErr} deg')
print(f'Frobenius norm of anglular velocity error: {velocityErr} deg/s')

mapping = [2, 3, 1, 0]

plt.figure(1)
plt.clf()
for pltState in range(dof):
    plt.subplot(2,2,pltState+1)
    plt.title(angles_ctrl[pltState])
    plt.plot(time, pred_ang[:, mapping[pltState]])
    plt.plot(time, np.degrees(qs[pltState,:]), 'r', label=f'2LayerTG')
    plt.plot(time, np.degrees(trajs[pltState,:]), 'm--', label=f'2Layer')
    
    plt.plot(time, np.degrees(trajsTGOnly[pltState,:]), 'b--', label=f'SoloTG')
    
    #plt.plot(time, np.degrees(qs[pltState+dof,:]*Ts), 'g', label=f'vel')
    #plt.plot(time, np.degrees(trajs[pltState+dof,:]), 'b--', label=f'velTraj')

plt.legend()
plt.show(block=False)
