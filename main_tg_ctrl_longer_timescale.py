#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys

from tools.ctrl_tools import ControlAndDynamics
from tools.trajgen_tools import TrajectoryGenerator, WalkingData
from tools.angle_functions import anglesTG, anglesCtrl, mapTG2Ctrl, \
                            ctrl_to_tg, tg_to_ctrl

# Usage: python3 main_tg_ctrl.py <leg>
################################################################################
# User-defined parameters
################################################################################
filename = '/home/lisa/Downloads/walk_sls_legs_11.pickle'
#filename = '/home/pierre/data/tuthill/models/models_sls/walk_sls_legs_11.pickle'

walkingSettings = [15, 0, 0] # walking, turning, flipping speeds (mm/s)

numTGSteps  = 200   # How many timesteps to run TG for
Ts          = 1/300 # Sampling time
ctrlSpeed   = Ts
ctrlTGComm  = 10 # How many steps elapse before talking to TG (as a multiple of Ts)

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
CD = ControlAndDynamics(leg, anglePen, drvPen[leg], inputPen, Ts)

angleTG2      = np.zeros((dofTG, numTGSteps))
drvTG2        = np.zeros((dofTG, numTGSteps))
phaseTG2      = np.zeros(numTGSteps)
angleTG2[:,0], drvTG2[:,0], phaseTG2[0] = angInit, drvInit, phaseInit

ys = np.zeros([CD._Nx, numTGSteps])
us = np.zeros([CD._Nu, numTGSteps])

# No disturbances (yet)
dist = np.zeros(CD._Nx)

for t in range(numTGSteps-1):
    if not (t % ctrlTGComm):
        ang = angleTG2[:,t] + ctrl_to_tg(ys[0:CD._Nu,t], legPos)
        drv = drvTG2[:,t] + ctrl_to_tg(ys[CD._Nu:,t]*CD._Ts, legPos)

        angleTG2[:,t+1], drvTG2[:,t+1], phaseTG2[t+1] = \
            TG.step_forward(ang, drv, phaseTG2[t], contexts[t])

        # Generate trajectory for future
        for tau in range(t+1, min(t+ctrlTGComm+1, numTGSteps-1)):
            angleTG2[:,tau+1], drvTG2[:,tau+1], phaseTG2[tau+1] = \
                TG.step_forward(angleTG2[:,tau], drvTG2[:,tau], phaseTG2[tau], contexts[tau])

    us[:,t], ys[:,t+1] = CD.step_forward(ys[:,t], angleTG2[:,t],
                         angleTG2[:,t+1], drvTG2[:,t], drvTG2[:,t+1], dist)

# True angle + derivatives
dof      = CD._Nu
angle2   = angleTG2 + ctrl_to_tg(ys[0:dof,:], legPos)
drv2     = drvTG2 + ctrl_to_tg(ys[dof:,:]*CD._Ts, legPos)
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

