#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys

from ctrl_tools import ControlAndDynamics
from trajgen_tools import TrajectoryGenerator
from angle_functions import anglesTG, anglesCtrl, mapTG2Ctrl, \
                            ctrl_to_tg, tg_to_ctrl

# Usage: python3 main_tg_ctrl.py <leg> <optional: basic>
################################################################################
# User-defined parameters
################################################################################
filename = '/home/lisa/Downloads/walk_sls_legs_8.pickle'
#filename = '/home/pierre/data/tuthill/models/models_sls/walk_sls_legs_8.pickle'

numTGSteps  = 200   # How many timesteps to run TG for
Ts          = 1/300 # Sampling time
ctrlTsRatio = 5    # Controller will sample at Ts / ctrlTsRatio

# LQR penalties
drvPen = {'L1': 1e-2, # Looks gait-like but different from pure TG
          'L2': 1e-2, # OK
          'L3': 1e-2, # OK
          'R1': 1e-2, # Looks gait-like but different from pure TG
          'R2': 1e-2, # A bit off pure TG
          'R3': 1e-2  # OK
         }
anglePen = 1e0
inputPen = 1e-8

leg = sys.argv[1]

# Use this option to track pre-generated trajectory
basicTracking = False  
if len(sys.argv) > 2 and sys.argv[2] == 'basic':
    print('Basic tracking (use trajectory from solo TG)')
    basicTracking = True

################################################################################
# Trajectory generator
################################################################################
legPos  = int(leg[-1])
dofTG   = len(anglesTG)
TG      = TrajectoryGenerator(filename, leg, dofTG, numTGSteps)

angleTG = np.zeros((dofTG, numTGSteps))
drvTG   = np.zeros((dofTG, numTGSteps))
phaseTG = np.zeros(numTGSteps)

angleTG[:,0], drvTG[:,0], phaseTG[0] = TG.get_initial_vals()

for t in range(numTGSteps-1):
    angleTG[:,t+1], drvTG[:,t+1], phaseTG[t+1] = \
        TG.step_forward(angleTG[:,t], drvTG[:,t], phaseTG[t], TG._context[t])

################################################################################
# Trajectory generator + ctrl and dynamics
################################################################################
numSimSteps = numTGSteps*ctrlTsRatio

CD    = ControlAndDynamics(leg, anglePen, drvPen[leg], inputPen, Ts/ctrlTsRatio)
dof   = CD._Nu
ys    = np.zeros([CD._Nx, numSimSteps])
us    = np.zeros([CD._Nu, numSimSteps])
dists = np.zeros([CD._Nx, numSimSteps]) # perturbations

if basicTracking:
    angleTG2 = angleTG
    drvTG2   = drvTG
    phaseTG2 = phaseTG
else:
    angleTG2 = np.zeros((dofTG, numTGSteps))
    drvTG2   = np.zeros((dofTG, numTGSteps))
    phaseTG2 = np.zeros(numTGSteps)
    ang, drv, phase = TG.get_initial_vals()
    angleTG2[:,0], drvTG2[:,0], phaseTG2[0] = ang, drv, phase

for t in range(numSimSteps-1):
    k  = int(t / ctrlTsRatio)      # Index for TG data
    kn = int((t+1) / ctrlTsRatio)  # Next index for TG data
    
    if not basicTracking and not ((t+1) % ctrlTsRatio): 
        # Only update TG if we are not doing basic tracking
        ang = angleTG2[:,k] + ctrl_to_tg(ys[0:dof,t], legPos)         
        drv = drvTG2[:,k] + ctrl_to_tg(ys[dof:,t]*CD._Ts, legPos)
        
        angleTG2[:,k+1], drvTG2[:,k+1], phaseTG2[k+1] = \
            TG.step_forward(ang, drv, phaseTG2[k], TG._context[k])

    us[:,t], ys[:,t+1] = CD.step_forward(ys[:,t], angleTG2[:,k], angleTG2[:,kn],
                                         drvTG2[:,k]/ctrlTsRatio, drvTG2[:,kn]/ctrlTsRatio, dists[:,t])

# True angle + derivative (sampled at Ts)
downSamp = list(range(ctrlTsRatio-1, numSimSteps, ctrlTsRatio))
angle2   = angleTG2 + ctrl_to_tg(ys[0:dof,downSamp], legPos)
drv2     = drvTG2 + ctrl_to_tg(ys[dof:,downSamp]*CD._Ts, legPos)
time     = np.array(range(numTGSteps))

plt.figure(1)
plt.clf()
for i in range(dof):
    plt.subplot(2,dof,i+1)
    plt.title(anglesCtrl[legPos][i])
    idx = mapTG2Ctrl[legPos][i]
    
    plt.plot(time, angleTG[idx,:], 'b', label=f'SoloTG')
    plt.plot(time, angleTG2[idx,:], 'r', label=f'2LayerTG')
    plt.plot(time, angle2[idx,:], 'k--', label=f'2Layer')
    
    plt.subplot(2,dof,i+dof+1)
    plt.title('Velocity')
    plt.plot(time, drvTG[idx,:], 'b', label=f'SoloTG')
    plt.plot(time, drvTG2[idx,:], 'r', label=f'2LayerTG')
    plt.plot(time, drv2[idx,:], 'k--', label=f'2Layer')
plt.legend()
plt.show()

