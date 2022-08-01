#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys

from tools.ctrl_tools import ControlAndDynamics
from tools.trajgen_tools import TrajectoryGenerator, WalkingData
from tools.angle_functions import anglesTG, anglesCtrl, mapTG2Ctrl, \
                            ctrl_to_tg, tg_to_ctrl

# Usage: python3 main_tg_ctrl.py <leg> <optional: basic>
################################################################################
# User-defined parameters
################################################################################
filename = '/home/lisa/Downloads/walk_sls_legs_11.pickle'
#filename = '/home/pierre/data/tuthill/models/models_sls/walk_sls_legs_11.pickle'

numTGSteps  = 200   # How many timesteps to run TG for
Ts          = 1/300 # Sampling time
ctrlTsRatio = 5    # Controller will sample at Ts / ctrlTsRatio

# LQR penalties
drvPen = {'L1': 1e-2, # Looks gait-like but different from pure TG
          'L2': 1e-2, # Good
          'L3': 1e-2, # Looks the same but slower than pure TG
          'R1': 1e-2, # Looks the same but slower than pure TG
          'R2': 1e-2, # I mean, even pure TG doesn't look right
          'R3': 1e-2  # Looks the same but slower than pure TG
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

wd       = WalkingData(filename)
bout     = wd.get_bout([15, 0, 0])
contexts = bout['contexts']

angleTG = np.zeros((dofTG, numTGSteps))
drvTG   = np.zeros((dofTG, numTGSteps))
phaseTG = np.zeros(numTGSteps)

angleTG[:,0] = bout['angles'][leg][0]
drvTG[:,0]   = bout['derivatives'][leg][0]
phaseTG[0]   = bout['phases'][leg][0]

for t in range(numTGSteps-1):
    angleTG[:,t+1], drvTG[:,t+1], phaseTG[t+1] = \
        TG.step_forward(angleTG[:,t], drvTG[:,t], phaseTG[t], contexts[t])

################################################################################
# Trajectory generator + ctrl and dynamics
################################################################################
numSimSteps = numTGSteps*ctrlTsRatio

CD    = ControlAndDynamics(leg, anglePen, drvPen[leg], inputPen, Ts/ctrlTsRatio)
dists = np.zeros([CD._Nx, numSimSteps])

if basicTracking:
    angleTG2 = angleTG
    drvTG2   = drvTG
    phaseTG2 = phaseTG
    ys       = CD.run_basic(angleTG2, drvTG2, ctrlTsRatio, dists)
else: 
    angleTG2, drvTG2, ys = CD.run(TG, contexts, numTGSteps, ctrlTsRatio, dists, bout)

# True angle + derivative (sampled at Ts)
dof      = CD._Nu
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

