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
numSimSteps = 200   # How many timesteps to run model for
Ts          = 1/300 # Sampling time
TGInterval  = 1     # Give feedback to TG once per interval 

# LQR penalties
drvPen = {'L1': 1e1, # Weird
          'L2': 1e0, # OK
          'L3': 1e0, # OK
          'R1': 1e1, # Weird
          'R2': 1e2, # Weird
          'R3': 1e0  # OK
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
filename = '/home/lisa/Downloads/walk_sls_legs_8.pickle'
#filename = '/home/pierre/data/tuthill/models/models_sls/walk_sls_legs_8.pickle'

legPos  = int(leg[-1])
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
# Trajectory generator + ctrl and dynamics
################################################################################
CD  = ControlAndDynamics(leg, anglePen, drvPen[leg], inputPen, Ts)
dof = CD._Nu
ys  = np.zeros([CD._Nx, numSimSteps])
us  = np.zeros([CD._Nu, numSimSteps])

if basicTracking:
    angleTG2 = angleTG
    drvTG2   = drvTG
    phaseTG2 = phaseTG
else:
    angleTG2 = np.zeros((dofTG, numSimSteps))
    drvTG2   = np.zeros((dofTG, numSimSteps))
    phaseTG2 = np.zeros(numSimSteps)
    ang, drv, phase = TG.get_initial_vals()
    angleTG2[:,0], drvTG2[:,0], phaseTG2[0] = ang, drv, phase

for t in range(numSimSteps-1):
    if not basicTracking: # when doing basic tracking, no need to update TG
        angleTG2[:,t+1], drvTG2[:,t+1], phaseTG2[t+1] = \
            TG.step_forward(ang, drv, phaseTG2[t], TG._context[t])
    
    us[:,t], ys[:,t+1] = CD.step_forward(ys[:,t], angleTG2[:,t], angleTG2[:,t+1],
                                         drvTG2[:,t], drvTG2[:,t+1])
    
    ang = angleTG2[:,t+1]   
    drv = drvTG2[:,t+1]

    if not ((t+1) % TGInterval):
        ang = ang + ctrl_to_tg(ys[0:dof,t+1], legPos) 
        drv = drv + ctrl_to_tg(ys[dof:,t+1]*Ts, legPos)

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

