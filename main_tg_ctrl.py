#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys

from tools.ctrl_tools import ControlAndDynamics
from tools.trajgen_tools import TrajectoryGenerator, WalkingData
from tools.angle_functions import anglesTG, anglesCtrl, mapTG2Ctrl, \
                            ctrl_to_tg, tg_to_ctrl

# Usage: python3 main_tg_ctrl_updated.py <leg>
################################################################################
# User-defined parameters
################################################################################
filename = '/home/lisa/Downloads/walk_sls_legs_11.pickle'
#filename = '/home/pierre/data/tuthill/models/models_sls/walk_sls_legs_11.pickle'

walkingSettings = [15, 0, 0] # walking, turning, flipping speeds (mm/s)

numTGSteps     = 200   # How many timesteps to run TG for
Ts             = 1/300 # How fast TG runs
ctrlSpeedRatio = 2     # Controller will run at Ts / ctrlSpeedRatio
ctrlCommRatio  = 8     # Controller communicates to TG this often (as multiple of Ts)
actDelay       = 0     # Seconds; typically 0.02-0.04

# LQR penalties
drvPen = {'L1': 1e-2, # 
          'L2': 1e-2, # 
          'L3': 1e-2, # 
          'R1': 1e-2, # 
          'R2': 1e-2, # 
          'R3': 1e-2  #
         }
anglePen = 1e0
inputPen = 1e-8

leg = sys.argv[1]

################################################################################
# Get walking data
################################################################################
wd       = WalkingData(filename)
bout     = wd.get_bout(walkingSettings)
contexts = bout['contexts']

angInit   = bout['angles'][leg][0]
drvInit   = bout['derivatives'][leg][0]
phaseInit = bout['phases'][leg][0]

################################################################################
# Solo trajectory generator
################################################################################
legPos  = int(leg[-1])
dofTG   = len(anglesTG)
TG      = TrajectoryGenerator(filename, leg, dofTG, numTGSteps)

angleTG = np.zeros((dofTG, numTGSteps))
drvTG   = np.zeros((dofTG, numTGSteps))
phaseTG = np.zeros(numTGSteps)
angleTG[:,0], drvTG[:,0], phaseTG[0] = angInit, drvInit, phaseInit

for t in range(numTGSteps-1):
    angleTG[:,t+1], drvTG[:,t+1], phaseTG[t+1] = \
        TG.step_forward(angleTG[:,t], drvTG[:,t], phaseTG[t], contexts[t])

################################################################################
# Trajectory generator + ctrl and dynamics
################################################################################
numDelays = int(actDelay / Ts * ctrlSpeedRatio)
print(f'Steps of actuation delay: {numDelays}')
CD        = ControlAndDynamics(leg, anglePen, drvPen[leg], inputPen, 
                               Ts/ctrlSpeedRatio, numDelays)

numSimSteps   = numTGSteps*ctrlSpeedRatio
angleTG2      = np.zeros((dofTG, numTGSteps))
drvTG2        = np.zeros((dofTG, numTGSteps))
phaseTG2      = np.zeros(numTGSteps)
angleTG2[:,0], drvTG2[:,0], phaseTG2[0] = angInit, drvInit, phaseInit

ys    = np.zeros([CD._Nx, numSimSteps])
us    = np.zeros([CD._Nu, numSimSteps])
dist  = np.zeros(CD._Nr) # Zero disturbances

for t in range(numSimSteps-1):
    k  = int(t / ctrlSpeedRatio)     # Index for TG data
    kn = int((t+1) / ctrlSpeedRatio) # Next index for TG data

    if not (k % ctrlCommRatio) and k != kn and k < numTGSteps-1:
        ang   = angleTG2[:,k] + ctrl_to_tg(ys[0:CD._Nu,t], legPos)
        drv   = drvTG2[:,k] + ctrl_to_tg(ys[CD._Nu:,t]*CD._Ts, legPos)
        angleTG2[:,k+1], drvTG2[:,k+1], phaseTG2[k+1] = \
            TG.step_forward(ang, drv, phaseTG2[k], contexts[k])

        # Generate trajectory for future
        for m in range(k+1, min(k+ctrlCommRatio, numTGSteps-1)):
            angleTG2[:,m+1], drvTG2[:,m+1], phaseTG2[m+1] = \
                TG.step_forward(angleTG2[:,m], drvTG2[:,m], phaseTG2[m], contexts[m])

    us[:,t], ys[:,t+1] = CD.step_forward(ys[:,t], angleTG2[:,k], angleTG2[:,kn], 
                         drvTG2[:,k]/ctrlSpeedRatio, drvTG2[:,kn]/ctrlSpeedRatio, dist)

# True angle + derivatives
dof      = CD._Nu
downSamp = list(range(ctrlSpeedRatio-1, numSimSteps, ctrlSpeedRatio))
angle2   = angleTG2 + ctrl_to_tg(ys[0:dof,downSamp], legPos)
drv2     = drvTG2 + ctrl_to_tg(ys[dof:CD._Nr,downSamp]*CD._Ts, legPos)
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

