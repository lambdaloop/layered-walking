#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys

from tools.ctrl_tools import ControlAndDynamics
from tools.trajgen_tools import TrajectoryGenerator, WalkingData
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

wd       = WalkingData(filename)
bout     = wd.get_bout([15, 0, 0])
contexts = bout['contexts']

ang   = bout['angles'][leg][0]
drv   = bout['derivatives'][leg][0]
phase = bout['phases'][leg][0]

numSimSteps = numTGSteps*ctrlTsRatio
CD          = ControlAndDynamics(leg, anglePen, drvPen[leg], inputPen, Ts/ctrlTsRatio)

# Simulate without disturbance (for comparison)
angleTG, drvTG, ys = CD.run(TG, contexts, numTGSteps, ctrlTsRatio, bout)

################################################################################
# Experimental: TG only, but with disturbances
################################################################################
dof        = CD._Nu

legIdx = legs.index(leg)
np.random.seed(600) # For perturbations generated randomly
angleTGPure = np.zeros((dofTG, numTGSteps))
drvTGPure   = np.zeros((dofTG, numTGSteps))
phaseTGPure = np.zeros(numTGSteps)

fullAngleNames = [(leg + ang) for ang in anglesTG]

'''
window = 2 # Amount of steps to look back and forward for minimum
heightsPure        = np.array([None] * numTGSteps)
groundContactPure = np.array([None] * numTGSteps)

angleTGPure[:,0], drvTGPure[:,0], phaseTGPure[0] = ang, drv, phase

for t in range(numTGSteps-1):
    heightsPure[t] = get_current_height(angleTGPure[:,t], fullAngleNames, legIdx)
    if t > window*2: # Live detection
        center = t-window
        if heightsPure[center] == min(heightsPure[center-window:t]): # center is minimum
            groundContactPure[t] = heightsPure[t]
            
            # TODO: hardcoded
            dist = get_dists_slippery(200)[leg]
            drvTGPure[:,t] = drvTGPure[:,t] + ctrl_to_tg(dist[dof:], legPos)*CD._Ts
            
    angleTGPure[:,t+1], drvTGPure[:,t+1], phaseTGPure[t+1] = \
        TG.step_forward(angleTGPure[:,t], drvTGPure[:,t], phaseTGPure[t], contexts[t])
'''

################################################################################
# Simulate with disturbances
################################################################################
np.random.seed(623) # For perturbations generated randomly

distType  = DistType.SLIPPERY_SURFACE
#distType = DistType.UNEVEN_SURFACE
#distType = DistType.BUMP_ON_SURFACE # OK for some, bad for others
#distType = DistType.SLOPED_SURFACE
#distType = DistType.MISSING_LEG  # This might correspond to too-large disturbance

# Contains params relevant to any type of disturbance
distDict = {'maxVelocity' : 200,        # Slippery surface
            'maxHt'       : 0.1/1000,   # Uneven surface
            'height'      : -0.1/1000,  # Stepping on a bump/pit
            'distLeg'     : leg,        # Stepping on a bump/pit
            'angle'       : 10,         # Walking on slope (degrees)
            'missingLeg'  : 'L1'        # Missing leg
           }
distDict['distType'] = distType

angleTGDist      = np.zeros((dofTG, numTGSteps))
drvTGDist        = np.zeros((dofTG, numTGSteps))
phaseTGDist      = np.zeros(numTGSteps)
angleTGDist[:,0], drvTGDist[:,0], phaseTGDist[0] = ang, drv, phase

ysDist = np.zeros([CD._Nx, numSimSteps])
usDist = np.zeros([CD._Nu, numSimSteps])

# Visualize height detection and compare heights
heightsDist     = np.array([None] * numSimSteps)
groundContact   = np.array([None] * numSimSteps)
locMinWindow    = 2*ctrlTsRatio
nonRepeatWindow = 3*ctrlTsRatio # Assumed minimum distance between minima
lastDetection   = -nonRepeatWindow

for t in range(numSimSteps-1):
    k  = int(t / ctrlTsRatio)      # Index for TG data
    kn = int((t+1) / ctrlTsRatio)  # Next index for TG data      
    
    ang = angleTGDist[:,k] + ctrl_to_tg(ysDist[0:CD._Nu,t], legPos)
    if not ((t+1) % ctrlTsRatio):             
        drv = drvTGDist[:,k] + ctrl_to_tg(ysDist[CD._Nu:,t]*CD._Ts, legPos)
        
        angleTGDist[:,k+1], drvTGDist[:,k+1], phaseTGDist[k+1] = \
            TG.step_forward(ang, drv, phaseTGDist[k], contexts[k])
    
    dist           = get_zero_dists()[leg]    
    heightsDist[t] = get_current_height(ang, fullAngleNames, legIdx)
    
    if loc_min_detected(locMinWindow, nonRepeatWindow, lastDetection, heightsDist, t):
        groundContact[t] = heightsDist[t] # Visualize height minimum detection
        lastDetection    = t
        dist             = get_dist(distDict, leg)               

    usDist[:,t], ysDist[:,t+1] = CD.step_forward(ysDist[:,t], angleTGDist[:,k],
        angleTGDist[:,kn], drvTGDist[:,k]/ctrlTsRatio, drvTGDist[:,kn]/ctrlTsRatio, dist)

# True angle + derivative (sampled at Ts)
downSamp   = list(range(ctrlTsRatio-1, numSimSteps, ctrlTsRatio))
angle2     = angleTG + ctrl_to_tg(ys[0:dof,downSamp], legPos)
drv2       = drvTG + ctrl_to_tg(ysDist[dof:,downSamp]*CD._Ts, legPos)
angle2Dist = angleTGDist + ctrl_to_tg(ysDist[0:dof,downSamp], legPos)
drv2Dist   = drvTGDist + ctrl_to_tg(ysDist[dof:,downSamp]*CD._Ts, legPos)

# Get heights for non-perturbed case as well
heights       = np.array([None] * numTGSteps)
for t in range(numTGSteps-1):
    heights[t] = get_current_height(angle2[:,t], fullAngleNames, legIdx)

time  = np.array(range(numTGSteps))
time2 = np.array(range(numSimSteps)) / ctrlTsRatio

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
    if i==0:
        plt.legend()
    plt.subplot(3,dof,i+dof+1)
    plt.title('Velocity')
    plt.plot(time, drvTG[idx,:], 'b', label=f'2LayerTG')
    plt.plot(time, drv2[idx,:], 'g--', label=f'2Layer')
    plt.plot(time, drvTGDist[idx,:], 'r', label=f'2LayerTG-Dist')
    plt.plot(time, drv2Dist[idx,:], 'm--', label=f'2Layer-Dist')
    
    plt.subplot(3,dof,i+2*dof+1)
    plt.title('Disturbance injected')
    plt.plot(time, heights, 'g')
    plt.plot(time2, heightsDist, 'm')
    plt.plot(time2, groundContact, 'k*')
    #plt.plot(time, groundContactPure, 'k*')

plt.show()


