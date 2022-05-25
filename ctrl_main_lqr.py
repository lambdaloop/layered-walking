#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ctrl_tools import *
from trajgen_tools import *
from angle_functions import angles_to_pose_names, make_fly_video

################################################################################
# User-defined parameters
################################################################################
leg         = 'L1'
numSimSteps = 200 # How many timesteps to run model for
Ts          = 1/300 # Sampling time

# LQR penalties
anglePen = 1e0
drvPen   = 5e1
inputPen = 1e-8

makeVideo = False

################################################################################
# Generate controller
################################################################################
CD = ControlAndDynamics(leg, anglePen, drvPen, inputPen, Ts)

################################################################################
# Solo TG
################################################################################
filename = '/home/lisa/Downloads/walk_sls_legs_2.pickle'
# filename = '/home/pierre/data/tuthill/models/models_sls/walk_sls_legs_2.pickle'

dofTG   = len(anglesTG)
TG      = TrajectoryGenerator(filename, leg, dofTG, numSimSteps)
angleTG = np.zeros((dofTG, numSimSteps))
drvTG   = np.zeros((dofTG, numSimSteps))
phaseTG = np.zeros(numSimSteps)

angleTG[:,0], drvTG[:,0], phaseTG[0] = TG.get_initial_vals() 

for t in range(numSimSteps-1):
    angleTG[:,t+1], drvTG[:,t+1], phaseTG[t+1] = \
        TG.step_forward(angleTG[:,t], drvTG[:,t], phaseTG[t], TG._context[t])

################################################################################
# 2-Layer: TG + Controller + Dynamics
################################################################################
legPos   = int(leg[-1])
dof      = CD._Nu
ys       = np.zeros([CD._Nx, numSimSteps])
us       = np.zeros([CD._Nu, numSimSteps])

angleTG2 = np.zeros((dofTG, numSimSteps))
drvTG2   = np.zeros((dofTG, numSimSteps))
phaseTG2 = np.zeros(numSimSteps)

ang, drv, phase = TG.get_initial_vals() 
angleTG2[:,0], drvTG2[:,0], phaseTG2[0] = ang, drv, phase

for t in range(numSimSteps-1):
    angleTG2[:,t+1], drvTG2[:,t+1], phaseTG2[t+1] = \
        TG.step_forward(ang, drv, phaseTG2[t], TG._context[t])
    
    us[:,t], ys[:,t+1] = CD.step_forward(ys[:,t], angleTG2[:,t], angleTG2[:,t+1],
                                         drvTG2[:,t], drvTG2[:,t+1])
    
    ang = angleTG2[:,t+1] + ctrl_to_tg(ys[0:dof,t+1], legPos)    
    drv = drvTG2[:,t+1]   + ctrl_to_tg(ys[dof:,t+1]*Ts, legPos)
    
angleErr = np.linalg.norm(np.degrees(ys[0:dof,]), ord='fro')
drvErr   = np.linalg.norm(np.degrees(ys[dof:,]*Ts), ord='fro')

print(f'Frob norm of angle error: {angleErr} deg')
print(f'Frob norm of angular velocity error: {drvErr} deg/s')

angle2 = tg_to_ctrl(angleTG2, legPos) + ys[0:dof,:]
drv2   = tg_to_ctrl(drvTG2, legPos)   + ys[dof:,:]*Ts
time   = np.array(range(numSimSteps))

plt.figure(1)
plt.clf()
for i in range(dof):
    plt.subplot(2,dof,i+1)
    plt.title(anglesCtrl[legPos][i])
    idx = mapTG2Ctrl[legPos][i]
    
    plt.plot(time, angleTG[idx,:], 'b', label=f'SoloTG')
    plt.plot(time, angleTG2[idx,:], 'r', label=f'2LayerTG')
    plt.plot(time, np.degrees(angle2[i,:]), 'k--', label=f'2Layer')
    
    plt.subplot(2,dof,i+dof+1)
    plt.title('Velocity')
    plt.plot(time, drvTG[idx,:], 'b', label=f'SoloTG')
    plt.plot(time, drvTG2[idx,:], 'r', label=f'2LayerTG')
    plt.plot(time, np.degrees(drv2[i,:]), 'k--', label=f'2Layer')
plt.legend()
plt.show()

if makeVideo:
    matplotlib.use('Agg')
    angs           = np.degrees(angle2).T
    angNames       = [(leg + ang) for ang in anglesCtrl[legPos]]
    pose_3d        = angles_to_pose_names(angs, angNames)
    pose_3d[:, 1:] = np.nan
    make_fly_video(pose_3d, 'vids/' + leg + '_twolayer.mp4')
