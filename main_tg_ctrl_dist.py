#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys

from tools.ctrl_tools import ControlAndDynamics
from tools.trajgen_tools import TrajectoryGenerator
from tools.angle_functions import anglesTG, anglesCtrl, mapTG2Ctrl, \
                            ctrl_to_tg, tg_to_ctrl, legs
from tools.dist_tools import *

# Usage: python3 main_tg_ctrl_dist.py <leg>
################################################################################
# User-defined parameters
################################################################################
filename = '/home/lisa/Downloads/walk_sls_legs_11.pickle'
#filename = '/home/pierre/data/tuthill/models/models_sls/walk_sls_legs_11.pickle'

numTGSteps  = 200   # How many timesteps to run TG for
Ts          = 1/300 # Sampling time
ctrlTsRatio = 5     # Controller will sample at Ts / ctrlTsRatio

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

################################################################################
# Trajectory generator + ctrl and dynamics
################################################################################
legPos  = int(leg[-1])
dofTG   = len(anglesTG)
TG      = TrajectoryGenerator(filename, leg, dofTG, numTGSteps)

numSimSteps = numTGSteps*ctrlTsRatio
CD          = ControlAndDynamics(leg, anglePen, drvPen[leg], inputPen, Ts/ctrlTsRatio)

# Simulate without disturbance (for comparison)
distsZero         = np.zeros([CD._Nx, numSimSteps])
angleTG, drvTG, ysTG = CD.run(TG, TG._context, numTGSteps, ctrlTsRatio, distsZero)

################################################################################
# Simulate with disturbances
################################################################################
legIdx = legs.index(leg)
np.random.seed(623) # For perturbations generated randomly

distType = DistType.SLIPPERY_SURFACE
    
# Slippery surface
maxVelocity = 100

# Uneven surface
maxHt     = 0.05/1000

# Stepping on a bump (+ve) or in a pit (-ve)
height    = -0.01/1000

# Walking on an incline (+ve) or decline (-ve)
angle     = -np.radians(30)

# Leg is missing
missingLeg = 'L1'

angleTGDist      = np.zeros((dofTG, numTGSteps))
drvTGDist        = np.zeros((dofTG, numTGSteps))
phaseTGDist      = np.zeros(numTGSteps)
ang, drv, phase = TG.get_initial_vals()
angleTGDist[:,0], drvTGDist[:,0], phaseTGDist[0] = ang, drv, phase

ysDist = np.zeros([CD._Nx, numSimSteps])
usDist = np.zeros([CD._Nu, numSimSteps])

# Visualize height detection
heights = np.array([None] * numSimSteps)
groundContact = np.array([None] * numSimSteps)
window = 2*ctrlTsRatio

fullAngleNames = [(leg + ang) for ang in anglesTG]

for t in range(numSimSteps-1):
    k  = int(t / ctrlTsRatio)      # Index for TG data
    kn = int((t+1) / ctrlTsRatio)  # Next index for TG data      
    
    ang = angleTGDist[:,k] + ctrl_to_tg(ysDist[0:CD._Nu,t], legPos)
    if not ((t+1) % ctrlTsRatio):             
        drv = drvTGDist[:,k] + ctrl_to_tg(ysDist[CD._Nu:,t]*CD._Ts, legPos)
        
        angleTGDist[:,k+1], drvTGDist[:,k+1], phaseTGDist[k+1] = \
            TG.step_forward(ang, drv, phaseTGDist[k], TG._context[k])
    
    dist = get_zero_dists()[leg]
    
    heights[t] = get_current_height(ang, fullAngleNames, legIdx)
    if t > window*2: # Live detection
        center = t-window
        if heights[center] == min(heights[center-window:t]): # center is minimum
            groundContact[t] = heights[t] # visualize height detection
            if distType == DistType.SLIPPERY_SURFACE:
                dist = get_dists_slippery(maxVelocity)[leg]
            elif distType == DistType.UNEVEN_SURFACE:
                dist = get_dists_uneven(maxHt)[leg]
            elif distType == DistType.BUMP_ON_SURFACE:
                dist = get_dists_bump_or_pit(height, distLeg)[leg]
            elif distType == DistType.SLOPED_SURFACE:
                dist = get_dists_incline_or_decline(angle)[leg]
            elif distType == DistType.MISSING_LEG:
                dist = get_dists_missing_leg(missingLeg)[leg]
            else:
                pass

    usDist[:,t], ysDist[:,t+1] = CD.step_forward(ysDist[:,t], angleTGDist[:,k], angleTGDist[:,kn],
        drvTGDist[:,k]/ctrlTsRatio, drvTGDist[:,kn]/ctrlTsRatio, dist)

# True angle + derivative (sampled at Ts)
dof        = CD._Nu
downSamp   = list(range(ctrlTsRatio-1, numSimSteps, ctrlTsRatio))
angle2     = angleTG + ctrl_to_tg(ysDist[0:dof,downSamp], legPos)
drv2       = drvTG + ctrl_to_tg(ysDist[dof:,downSamp]*CD._Ts, legPos)
angle2Dist = angleTGDist + ctrl_to_tg(ysDist[0:dof,downSamp], legPos)
drv2Dist   = drvTGDist + ctrl_to_tg(ysDist[dof:,downSamp]*CD._Ts, legPos)

time = np.array(range(numTGSteps))
time2 = np.array(range(numSimSteps))

plt.figure(1)
plt.clf()
for i in range(dof):
    plt.subplot(3,dof,i+1)
    plt.title(anglesCtrl[legPos][i])
    idx = mapTG2Ctrl[legPos][i]
    
    plt.plot(time, angleTG[idx,:], 'b', label=f'2LayerTG')
    plt.plot(time, angle2[idx,:], 'g--', label=f'2Layer')
    plt.plot(time, angleTGDist[idx,:], 'r', label=f'2LayerTG-Dist')
    plt.plot(time, angle2Dist[idx,:], 'm--', label=f'2Layer-Dist')
    
    plt.subplot(3,dof,i+dof+1)
    plt.title('Velocity')
    plt.plot(time, drvTG[idx,:], 'b', label=f'2LayerTG')
    plt.plot(time, drv2[idx,:], 'g--', label=f'2Layer')
    plt.plot(time, drvTGDist[idx,:], 'r', label=f'2LayerTG-Dist')
    plt.plot(time, drv2Dist[idx,:], 'm--', label=f'2Layer-Dist')
    plt.legend()
    
    plt.subplot(3,dof,2*dof+1)
    plt.title('Ground contact detection')
    plt.plot(time2, heights, 'b')
    plt.plot(time2, groundContact, 'k*')

plt.show()


