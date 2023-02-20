#!/usr/bin/env python

import math
import matplotlib.pyplot as plt
import numpy as np
import sys

from tools.ctrl_tools import ControlAndDynamics
from tools.trajgen_tools import TrajectoryGenerator, WalkingData
from tools.angle_functions import anglesTG, anglesCtrl, \
                                  ctrl_to_tg, tg_to_ctrl, legs
from tools.dist_tools import *

# Usage: python3 main_tg_ctrl_dist.py <leg>
################################################################################
# User-defined parameters
################################################################################
filename = '/home/lisa/Downloads/walk_sls_legs_subang_1.pickle'
#filename = '/home/pierre/data/tuthill/models/models_sls/walk_sls_legs_11.pickle'

walkingSettings = [15, 0, 0] # walking, turning, flipping speeds (mm/s)

numTGSteps      = 200   # How many timesteps to run TG for
Ts              = 1/300 # How fast TG runs
ctrlSpeedRatio  = 2     # Controller will run at Ts / ctrlSpeedRatio
ctrlCommRatio   = 8     # Controller communicates to TG this often (as multiple of Ts)
actDelay        = 0.03  # Seconds; typically 0.02-0.04
senseDelay      = 0.01  # Seconds; typically 0.01

lookaheadSTD = 0.02 # Standard deviation of gaussian noise
leg = sys.argv[1]

################################################################################
# Disturbance
################################################################################
distType = DistType.SLIPPERY_SURFACE
distDict = {'maxVelocity' : 1}
distDict['distType'] = distType

################################################################################
# Get walking data
################################################################################
wd       = WalkingData(filename)
bout     = wd.get_bout(walkingSettings)

# Use constant contexts
context  = np.array(walkingSettings).reshape(1,3)
contexts = np.repeat(context, numTGSteps, axis=0)

angInit   = bout['angles'][leg][0]
drvInit   = bout['derivatives'][leg][0]
phaseInit = bout['phases'][leg][0]

################################################################################
# Trajectory generator + ctrl and dynamics, w/o disturbances
################################################################################
dAct   = int(actDelay / Ts * ctrlSpeedRatio)
dSense = int(senseDelay / Ts * ctrlSpeedRatio)
print(f'Steps of actuation delay: {dAct}')
print(f'Steps of sensory delay  : {dSense}')

legPos  = int(leg[-1])
TG      = TrajectoryGenerator(filename, leg, numTGSteps)
numAng  = TG._numAng

namesTG = [x[2:] for x in TG._angle_names]
CD      = ControlAndDynamics(leg, Ts/ctrlSpeedRatio, dSense, dAct, 
                             namesTG=namesTG, lookaheadSTD = lookaheadSTD)

legIdx         = legs.index(leg)
fullAngleNames = [(leg + ang) for ang in namesTG]
dof            = CD._Nur

numSimSteps   = numTGSteps*ctrlSpeedRatio
angleTG2      = np.zeros((numAng, numTGSteps))
drvTG2        = np.zeros((numAng, numTGSteps))
phaseTG2      = np.zeros(numTGSteps)
angleTG2[:numAng,0], drvTG2[:numAng,0], phaseTG2[0] = angInit, drvInit, phaseInit

xs    = np.zeros([CD._Nx, numSimSteps])
xEsts = np.zeros([CD._Nx, numSimSteps])
us    = np.zeros([CD._Nu, numSimSteps])
dist  = np.zeros(CD._Nxr) # Zero disturbances

lookahead = math.ceil(dAct/ctrlSpeedRatio)

for t in range(numSimSteps-1):
    k  = int(t / ctrlSpeedRatio)     # Index for TG data
    kn = int((t+1) / ctrlSpeedRatio) # Next index for TG data

    if not (k % ctrlCommRatio) and k != kn and k < numTGSteps-1:
        ang   = angleTG2[:numAng,k] + ctrl_to_tg(xs[0:dof,t], legPos, namesTG)
        drv   = drvTG2[:numAng,k] + ctrl_to_tg(xs[dof:dof*2,t]*CD._Ts, legPos, namesTG)
        
        kEnd = min(k+ctrlCommRatio+lookahead, numTGSteps-1)
        angleTG2[:numAng,k+1:kEnd+1], drvTG2[:numAng,k+1:kEnd+1], phaseTG2[k+1:kEnd+1] = \
            TG.get_future_traj(k, kEnd, ang, drv, phaseTG2[k], contexts)

    k1 = min(int((t+dAct) / ctrlSpeedRatio), numTGSteps-1)
    k2 = min(int((t+dAct+1) / ctrlSpeedRatio), numTGSteps-1)
    
    anglesAhead = np.concatenate((angleTG2[:,k1].reshape(numAng,1),
                                  angleTG2[:,k2].reshape(numAng,1)), axis=1)
    drvsAhead   = np.concatenate((drvTG2[:,k1].reshape(numAng,1),
                                  drvTG2[:,k2].reshape(numAng,1)), axis=1)/ctrlSpeedRatio
    us[:,t], xs[:,t+1], xEsts[:,t+1] = CD.step_forward(xs[:,t], xEsts[:,t], anglesAhead, drvsAhead, dist)

################################################################################
# Simulate with disturbances
################################################################################
angleTGDist      = np.zeros((numAng, numTGSteps))
drvTGDist        = np.zeros((numAng, numTGSteps))
phaseTGDist      = np.zeros(numTGSteps)
angleTGDist[:,0], drvTGDist[:,0], phaseTGDist[0] = angInit, drvInit, phaseInit

xsDist    = np.zeros([CD._Nx, numSimSteps])
xEstsDist = np.zeros([CD._Nx, numSimSteps])
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
    ang = angleTGDist[:,k] + ctrl_to_tg(xsDist[0:dof,t], legPos, namesTG)
    drv = drvTGDist[:,k] + ctrl_to_tg(xsDist[dof:dof*2,t]*CD._Ts, legPos, namesTG)

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

    k1 = min(int((t+dAct) / ctrlSpeedRatio), numTGSteps-1)
    k2 = min(int((t+dAct+1) / ctrlSpeedRatio), numTGSteps-1)
    
    anglesAhead = np.concatenate((angleTGDist[:,k1].reshape(numAng,1),
                                  angleTGDist[:,k2].reshape(numAng,1)), axis=1)
    drvsAhead   = np.concatenate((drvTGDist[:,k1].reshape(numAng,1),
                                  drvTGDist[:,k2].reshape(numAng,1)), axis=1)/ctrlSpeedRatio
    
    usDist[:,t], xsDist[:,t+1], xEstsDist[:,t+1] = \
        CD.step_forward(xsDist[:,t], xEstsDist[:,t], anglesAhead, drvsAhead, dist)

################################################################################
# Post-processing for plotting
################################################################################
# True angle + derivative (sampled at Ts)
downSamp   = list(range(ctrlSpeedRatio-1, numSimSteps, ctrlSpeedRatio))
angle2     = angleTG2 + ctrl_to_tg(xs[0:dof,downSamp], legPos, namesTG)
drv2       = drvTG2 + ctrl_to_tg(xs[dof:dof*2,downSamp]*CD._Ts, legPos, namesTG)
angle2Dist = angleTGDist + ctrl_to_tg(xsDist[0:dof,downSamp], legPos, namesTG)
drv2Dist   = drvTGDist + ctrl_to_tg(xsDist[dof:dof*2,downSamp]*CD._Ts, legPos, namesTG)

# Get heights for non-perturbed case as well
heights       = np.array([None] * numTGSteps)
for t in range(numTGSteps-1):
    heights[t] = get_current_height(angle2[:,t], fullAngleNames, legIdx)

time  = np.array(range(numTGSteps))
time2 = np.array(range(numSimSteps)) / ctrlSpeedRatio

plt.figure(1)
plt.clf()

mapIx = [namesTG.index(n) for n in anglesCtrl[legPos]]

for i in range(dof):
    plt.subplot(3,dof,i+1)
    plt.title(anglesCtrl[legPos][i])
    idx = mapIx[i]
        
    plt.plot(time, angleTG2[idx,:], 'b', label=f'2LayerTG')
    plt.plot(time, angle2[idx,:], 'g--', label=f'2Layer')
    plt.plot(time, angleTGDist[idx,:], 'r', label=f'2LayerTG-Dist')
    plt.plot(time, angle2Dist[idx,:], 'm--', label=f'2Layer-Dist')

    if i==0:
        plt.legend()

    plt.subplot(3,dof,i+dof+1)
    plt.title('Velocity')
    plt.plot(time, drvTG2[idx,:], 'b', label=f'2LayerTG')
    plt.plot(time, drv2[idx,:], 'g--', label=f'2Layer')
    plt.plot(time, drvTGDist[idx,:], 'r', label=f'2LayerTG-Dist')
    plt.plot(time, drv2Dist[idx,:], 'm--', label=f'2Layer-Dist')
        
    plt.subplot(3,dof,i+2*dof+1)
    plt.title('Disturbance injected')
    plt.plot(time, heights, 'g', label=f'2Layer')
    plt.plot(time2, heightsDist, 'm', label=f'2Layer-Dist')
    
    plt.plot(time2, groundContactDist, 'r*', markersize=10)

plt.show()

