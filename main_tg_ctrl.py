#!/usr/bin/env python

import math
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

walkingSettings = [15, 0, 0] # walking, turning, flipping speeds (mm/s)

numTGSteps     = 200   # How many timesteps to run TG for
Ts             = 1/300 # How fast TG runs
ctrlSpeedRatio = 2     # Controller will run at Ts / ctrlSpeedRatio
ctrlCommRatio  = 8     # Controller communicates to TG this often (as multiple of Ts)
actDelay       = 0.01  # Seconds; typically 0.02-0.04

# LQR penalties
drvPen = {'L1': 1e-5, # 
          'L2': 1e-5, # 
          'L3': 1e-3, # 
          'R1': 1e-5, # 
          'R2': 1e-5, # 
          'R3': 1e-5  #
         }

futurePenRatio = 1.0 # y_hat(t+1) is penalized (ratio)*pen as much as y(t)
                     # y_hat(t+2) is penalized (ratio^2)*pen as much as y(t)
anglePen       = 1e0
inputPen       = 1e-8

leg = sys.argv[1]

################################################################################
# Disturbance
################################################################################
distType = DistType.ZERO
#distType = DistType.SLIPPERY_SURFACE
#distType = DistType.UNEVEN_SURFACE
#distType = DistType.BUMP_ON_SURFACE # OK for some, bad for others
#distType = DistType.SLOPED_SURFACE
#distType = DistType.MISSING_LEG  # This might correspond to too-large disturbance

# Contains params relevant to any type of disturbance
distDict = {'maxVelocity' : 50,        # Slippery surface
            'maxHt'       : 0.1/1000,   # Uneven surface
            'height'      : -0.1/1000,  # Stepping on a bump/pit
            'distLeg'     : leg,        # Stepping on a bump/pit
            'angle'       : 10,         # Walking on slope (degrees)
            'missingLeg'  : 'L1'        # Missing leg
           }
distDict['distType'] = distType

plotPureTG = False # Whether or not to also plot data from disturbed pure TG

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
# Trajectory generator + ctrl and dynamics, w/o disturbances
################################################################################
numDelays = int(actDelay / Ts * ctrlSpeedRatio)
print(f'Steps of actuation delay: {numDelays}')
CD        = ControlAndDynamics(leg, Ts/ctrlSpeedRatio, numDelays, futurePenRatio,
                               anglePen, drvPen[leg], inputPen)

legPos  = int(leg[-1])
dofTG   = len(anglesTG)
TG      = TrajectoryGenerator(filename, leg, dofTG, numTGSteps)

numSimSteps   = numTGSteps*ctrlSpeedRatio
angleTG2      = np.zeros((dofTG, numTGSteps))
drvTG2        = np.zeros((dofTG, numTGSteps))
phaseTG2      = np.zeros(numTGSteps)
angleTG2[:,0], drvTG2[:,0], phaseTG2[0] = angInit, drvInit, phaseInit

ys    = np.zeros([CD._Nx, numSimSteps])
us    = np.zeros([CD._Nu, numSimSteps])
dist  = np.zeros(CD._Nr) # Zero disturbances

lookahead = math.ceil(numDelays/ctrlSpeedRatio)

for t in range(numSimSteps-1):
    k  = int(t / ctrlSpeedRatio)     # Index for TG data
    kn = int((t+1) / ctrlSpeedRatio) # Next index for TG data

    if not (k % ctrlCommRatio) and k != kn and k < numTGSteps-1:
        ang   = angleTG2[:,k] + ctrl_to_tg(ys[0:CD._Nu,t], legPos)
        drv   = drvTG2[:,k] + ctrl_to_tg(ys[CD._Nu:,t]*CD._Ts, legPos)
        
        kEnd = min(k+ctrlCommRatio+lookahead, numTGSteps-1)
        angleTG2[:,k+1:kEnd+1], drvTG2[:,k+1:kEnd+1], phaseTG2[k+1:kEnd+1] = \
            TG.get_future_traj(k, kEnd, ang, drv, phaseTG2[k], contexts)

    k1 = min(int((t+numDelays) / ctrlSpeedRatio), numTGSteps-1)
    k2 = min(int((t+numDelays+1) / ctrlSpeedRatio), numTGSteps-1)
    
    anglesAhead = np.concatenate((angleTG2[:,k1].reshape(dofTG,1),
                                  angleTG2[:,k2].reshape(dofTG,1)), axis=1)
    drvsAhead   = np.concatenate((drvTG2[:,k1].reshape(dofTG,1),
                                  drvTG2[:,k2].reshape(dofTG,1)), axis=1)/ctrlSpeedRatio
        
    us[:,t], ys[:,t+1] = CD.step_forward(ys[:,t], anglesAhead, drvsAhead, dist)

################################################################################
# Experimental: Pure TG, but with disturbances
################################################################################
np.random.seed(600) # For perturbations generated randomly

dof            = CD._Nu
legIdx         = legs.index(leg)
fullAngleNames = [(leg + ang) for ang in anglesTG]

angleTGPure = np.zeros((dofTG, numTGSteps))
drvTGPure   = np.zeros((dofTG, numTGSteps))
phaseTGPure = np.zeros(numTGSteps)

locMinWindow    = 2
nonRepeatWindow = 3 # Assumed minimum distance between minima
lastDetection   = -nonRepeatWindow

heightsPure       = np.array([None] * numTGSteps)
groundContactPure = np.array([None] * numTGSteps)

angleTGPure[:,0], drvTGPure[:,0], phaseTGPure[0] = angInit, drvInit, phaseInit

for t in range(numTGSteps-1):        
    heightsPure[t] = get_current_height(angleTGPure[:,t], fullAngleNames, legIdx)

    if loc_min_detected(locMinWindow, nonRepeatWindow, lastDetection, heightsPure, t):
        groundContactPure[t] = heightsPure[t] # Visualize height minimum detection
        lastDetection        = t
        dist                 = get_dist(distDict, leg)       
        
        # Add perturbation
        angleTGPure[:,t] = angleTGPure[:,t] + ctrl_to_tg(dist[:dof], legPos)       
        drvTGPure[:,t]   = drvTGPure[:,t] + ctrl_to_tg(dist[dof:], legPos)*CD._Ts
            
    angleTGPure[:,t+1], drvTGPure[:,t+1], phaseTGPure[t+1] = \
        TG.step_forward(angleTGPure[:,t], drvTGPure[:,t], phaseTGPure[t], contexts[t])

################################################################################
# Simulate with disturbances
################################################################################
angleTGDist      = np.zeros((dofTG, numTGSteps))
drvTGDist        = np.zeros((dofTG, numTGSteps))
phaseTGDist      = np.zeros(numTGSteps)
angleTGDist[:,0], drvTGDist[:,0], phaseTGDist[0] = angInit, drvInit, phaseInit

ysDist    = np.zeros([CD._Nx, numSimSteps])
usDist    = np.zeros([CD._Nu, numSimSteps])

# Visualize height detection and compare heights
heightsDist       = np.array([None] * numSimSteps)
groundContactDist = np.array([None] * numSimSteps)
locMinWindow      = 2*ctrlSpeedRatio
nonRepeatWindow   = 10*ctrlSpeedRatio # Assumed minimum distance between minima
lastDetection     = -nonRepeatWindow

for t in range(numSimSteps-1):
    k   = int(t / ctrlSpeedRatio)     # Index for TG data
    kn  = int((t+1) / ctrlSpeedRatio) # Next index for TG data
    ang = angleTGDist[:,k] + ctrl_to_tg(ysDist[0:CD._Nu,t], legPos)
    drv = drvTGDist[:,k] + ctrl_to_tg(ysDist[CD._Nu:,t]*CD._Ts, legPos)

    if not (k % ctrlCommRatio) and k != kn and k < numTGSteps-1:        
        kEnd = min(k+ctrlCommRatio+lookahead, numTGSteps-1)
        angleTGDist[:,k+1:kEnd+1], drvTGDist[:,k+1:kEnd+1], phaseTGDist[k+1:kEnd+1] = \
            TG.get_future_traj(k, kEnd, ang, drv, phaseTGDist[k], contexts)
    
    dist           = get_zero_dists()[leg]    
    heightsDist[t] = get_current_height(ang, fullAngleNames, legIdx)
    
    if loc_min_detected(locMinWindow, nonRepeatWindow, lastDetection, heightsDist, t):
        groundContactDist[t] = heightsDist[t] # Visualize height minimum detection
        lastDetection        = t
        dist                 = get_dist(distDict, leg)               

    k1 = min(int((t+numDelays) / ctrlSpeedRatio), numTGSteps-1)
    k2 = min(int((t+numDelays+1) / ctrlSpeedRatio), numTGSteps-1)
    
    anglesAhead = np.concatenate((angleTGDist[:,k1].reshape(dofTG,1),
                                  angleTGDist[:,k2].reshape(dofTG,1)), axis=1)
    drvsAhead   = np.concatenate((drvTGDist[:,k1].reshape(dofTG,1),
                                  drvTGDist[:,k2].reshape(dofTG,1)), axis=1)/ctrlSpeedRatio
        
    usDist[:,t], ysDist[:,t+1] = CD.step_forward(ysDist[:,t], anglesAhead, drvsAhead, dist)

################################################################################
# Post-processing for plotting
################################################################################
# True angle + derivative (sampled at Ts)
downSamp   = list(range(ctrlSpeedRatio-1, numSimSteps, ctrlSpeedRatio))
angle2     = angleTG2 + ctrl_to_tg(ys[0:dof,downSamp], legPos)
drv2       = drvTG2 + ctrl_to_tg(ys[dof:,downSamp]*CD._Ts, legPos)
angle2Dist = angleTGDist + ctrl_to_tg(ysDist[0:dof,downSamp], legPos)
drv2Dist   = drvTGDist + ctrl_to_tg(ysDist[dof:,downSamp]*CD._Ts, legPos)

# Get heights for non-perturbed case as well
heights       = np.array([None] * numTGSteps)
for t in range(numTGSteps-1):
    heights[t] = get_current_height(angle2[:,t], fullAngleNames, legIdx)

time  = np.array(range(numTGSteps))
time2 = np.array(range(numSimSteps)) / ctrlSpeedRatio

plt.figure(1)
plt.clf()
for i in range(dof):
    plt.subplot(3,dof,i+1)
    plt.title(anglesCtrl[legPos][i])
    idx = mapTG2Ctrl[legPos][i]
    
    plt.plot(time, angleTG2[idx,:], 'b', label=f'2LayerTG')
    plt.plot(time, angle2[idx,:], 'g--', label=f'2Layer')
    plt.plot(time, angleTGDist[idx,:], 'r', label=f'2LayerTG-Dist')
    plt.plot(time, angle2Dist[idx,:], 'm--', label=f'2Layer-Dist')
    if plotPureTG:
        plt.plot(time, angleTGPure[idx,:], 'k', label=f'PureTG-Dist')

    if i==0:
        plt.legend()

    plt.subplot(3,dof,i+dof+1)
    plt.title('Velocity')
    plt.plot(time, drvTG2[idx,:], 'b', label=f'2LayerTG')
    plt.plot(time, drv2[idx,:], 'g--', label=f'2Layer')
    plt.plot(time, drvTGDist[idx,:], 'r', label=f'2LayerTG-Dist')
    plt.plot(time, drv2Dist[idx,:], 'm--', label=f'2Layer-Dist')
    if plotPureTG:
        plt.plot(time, drvTGPure[idx,:], 'k', label=f'PureTG-Dist')
        
    plt.subplot(3,dof,i+2*dof+1)
    plt.title('Disturbance injected')
    plt.plot(time, heights, 'g', label=f'2Layer')
    plt.plot(time2, heightsDist, 'm', label=f'2Layer-Dist')
    if plotPureTG:
        plt.plot(time, heightsPure, 'k', label=f'PureTG-Dist')
    
    plt.plot(time2, groundContactDist, 'r*', markersize=10)
    if plotPureTG:
        plt.plot(time, groundContactPure, 'k*')

plt.show()

