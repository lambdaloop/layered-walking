#!/usr/bin/env python

import control
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag
from ctrl_tools import *

# Sampling time
Ts = 1/300 # 300 times per second

# Assumes we already saved the linearized system in the appropriate files
# by running getLinearizedSystem()
ALin, BLin, xEqm, uEqm = loadLinearizedSystem()

dof = int(len(ALin)/2)
Nx  = 2*dof
Nu  = dof

# Discretize
A = np.eye(Nx) + ALin*Ts
B = BLin*Ts

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

K         = control.dlqr(A, B, Q, R)[0]
ACL       = A - B @ K

# Sanity check: stability
eigsCL    = np.linalg.eig(ACL)[0]
specRadCL = max(np.abs(eigsCL))
if specRadCL >= 1:
    print('Error: Controller did not stabilize!')

tHorizon = 100
time = np.array(range(tHorizon))
ys   = np.zeros([Nx, tHorizon])
us   = np.zeros([Nu, tHorizon])

'''
import tensorflow as tf
import pickle

fn = '/home/lisa/Downloads/walk_sls_legs_1.pickle'
with open(fn, 'rb') as f:
    allmodels = pickle.load(f)

from model_functions import MLPScaledXY

model_walk = MLPScaledXY.from_full(allmodels['L1']['model_walk'])
xy_w, bnums = allmodels['L1']['train']

n_ang = len(angles_main)
common = Counter(bnums).most_common(100)
b, _ = common[50]

n_pred = 200 # time horizon
cc = np.where(b == bnums)[0][:n_pred]
preds = []

# Around line 528 (single leg)
# "Real" is the actual trajectory, "pred" is that
# Want: TG vs TG+Dyn
# 800 (6 legs) onward from the .org file is stuff we want
angles_main = ['L1C_flex', 'L1A_abduct', 'L1B_flex', 'L1B_rot']
'''

# Trajectory is a sine wave
trajs      = np.zeros([Nx, tHorizon])
for joint in range(dof):
    trajs[joint,:] = 0.5 * np.sin(time * 0.25) + xEqm[joint]

for t in range(tHorizon-1):
    wtraj        = A @ (trajs[:,t] - xEqm) + xEqm - trajs[:,t+1]    
    ys[:,t+1]    = ACL @ ys[:,t] + wtraj
    us[:,t]      = K @ ys[:,t]

qs = ys + trajs

for pltState in range(dof):
    plt.subplot(2,2,pltState+1)
    plt.plot(time, np.degrees(qs[pltState,:]), '--', label=f'q{pltState}')
    plt.plot(time+2, np.degrees(trajs[pltState,:]), label=f'traj{pltState}')

plt.legend()
plt.show()


