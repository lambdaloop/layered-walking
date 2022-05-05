#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress warnings

import control
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag

from ctrl_tools import *
from trajgen_tools import *
from angle_functions import angles_to_pose_names, make_fly_video

numSimSteps = 600 # How many timesteps to run model for
time        = np.array(range(numSimSteps))

#################################################
# Generate discrete-time system and controller
#################################################
Ts = 1/300 # Sampling time

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
print(f'Open-loop spectral radius: {specRadOL}')

# Sanity check: controllability
Qc = control.ctrb(A,B)
rankCtrb = np.linalg.matrix_rank(Qc)
if rankCtrb != Nx:
    print('Error: System uncontrollable!')

# Controller objective
anglePenalty    = 1
velocityPenalty = 1e-8
inputPenalty    = 1e-8

Q1 = anglePenalty*np.eye(dof)
Q2 = velocityPenalty*np.eye(dof)
Q  = block_diag(Q1, Q2)
R  = inputPenalty*np.eye(Nu)

K   = control.dlqr(A, B, Q, R)[0]
ACL = A - B @ K
eigsCL    = np.linalg.eig(ACL)[0]
specRadCL = max(np.abs(eigsCL))
print(f'Closed-loop spectral radius: {specRadCL}')

#################################################
# Solo TG
#################################################
filename = '/home/lisa/Downloads/walk_sls_legs_2.pickle'
# filename = '/home/pierre/data/tuthill/models/models_sls/walk_sls_legs_2.pickle'
leg      = 'L1'
numAng   = 5

TG      = TrajectoryGenerator(filename, leg, numAng, numSimSteps)
angleTG = np.zeros((numAng, numSimSteps))
drvTG   = np.zeros((numAng, numSimSteps))
phaseTG = np.zeros(numSimSteps)

angleTG[:,0], drvTG[:,0], phaseTG[0] = TG.get_initial_vals() 

for t in range(numSimSteps-1):
    angleTG[:,t+1], drvTG[:,t+1], phaseTG[t+1] = \
        TG.step_forward(angleTG[:,t], drvTG[:,t], phaseTG[t], TG._context[t])

#################################################
# 2-Layer: TG + Controller + Dynamics
#################################################
xEqmFlat = xEqm.flatten()
ys       = np.zeros([Nx, numSimSteps])
us       = np.zeros([Nu, numSimSteps])

angleTG2 = np.zeros((numAng, numSimSteps))
drvTG2   = np.zeros((numAng, numSimSteps))
phaseTG2 = np.zeros(numSimSteps)

ang, drv, phase = TG.get_initial_vals() 
angleTG2[:,0]   = ang
drvTG2[:,0]     = drv
phaseTG2[0]     = phase

for t in range(numSimSteps-1):
    angleTG2[:,t+1], drvTG2[:,t+1], phaseTG2[t+1] = \
        TG.step_forward(ang, drv, phaseTG2[t], TG._context[t])
    
    trajNow  = np.append(tg_to_ctrl(angleTG2[:,t]), tg_to_ctrl(drvTG2[:,t]/Ts))
    trajNext = np.append(tg_to_ctrl(angleTG2[:,t+1]), tg_to_ctrl(drvTG2[:,t+1]/Ts))
    wtraj    = A @ (trajNow - xEqmFlat) + xEqmFlat - trajNext
    
    # Give some look-ahead to wtraj
    us[:,t]   = -K @ (ys[:,t] + wtraj)    
    ys[:,t+1] = A @ ys[:,t] + B @ us[:,t] + wtraj
    
    # For next step
    ang = angleTG2[:,t+1] + ctrl_to_tg(ys[0:dof,t+1], 0)
    
    drv = drvTG2[:,t+1] # TODO: need this + ctrl_to_tg(ys[dof:,t+1]*Ts, 0)    

angle2 = tg_to_ctrl(angleTG2) + ys[0:dof,:]
drv2   = tg_to_ctrl(drvTG2) + np.degrees(ys[dof:,:]*Ts)

angleErr    = np.linalg.norm(np.degrees(ys[0:dof,]), ord='fro')
velocityErr = np.linalg.norm(np.degrees(ys[dof:,]*Ts), ord='fro')

print(f'Frobenius norm of angle error: {angleErr} deg')
print(f'Frobenius norm of anglular velocity error: {velocityErr} deg/s')

plt.figure(1)
plt.clf()
for i in range(dof):
    plt.subplot(2,2,i+1)
    plt.title(anglesCtrl[i])
    idx = mapping[i]
    
    plt.plot(time, angleTG[idx,:], 'b', label=f'SoloTG')
    plt.plot(time, angleTG2[idx,:], 'r', label=f'2LayerTG')
    plt.plot(time, np.degrees(angle2[i,:]), 'm--', label=f'2Layer')        

plt.legend()
plt.show(block=False)

import matplotlib
matplotlib.use('Agg')

angs = np.degrees(angle2).T
pose_3d = angles_to_pose_names(angs, anglesCtrl)
pose_3d[:, 1:] = np.nan
make_fly_video(pose_3d, "vids/test_control.mp4")
