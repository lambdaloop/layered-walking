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
from tools.ground_model import GroundModel

# Usage: python3 main_tg_ctrl_dist.py <leg>
################################################################################
# User-defined parameters
################################################################################
filename = '/home/lisa/Downloads/walk_sls_legs_subang_1.pickle'

walkingSettings = [10, 0, 0] # walking, turning, flipping speeds (mm/s)

numTGSteps      = 200   # How many timesteps to run TG for
Ts              = 1/300 # How fast TG runs
ctrlSpeedRatio  = 2     # Controller will run at Ts / ctrlSpeedRatio
ctrlCommRatio   = 8     # Controller communicates to TG this often (as multiple of Ts)
actDelay        = 0.03  # Seconds; typically 0.02-0.04

leg = 'R1'

################################################################################
# Ground model
################################################################################
offset    = 3
startIdx  = 19
gndHeight = -0.85
ground    = GroundModel(offset=[0, 0, gndHeight], phi=0, theta=0)

################################################################################
# Get walking data
################################################################################
wd       = WalkingData(filename)
bout     = wd.get_bout(walkingSettings, offset=offset)

# Use constant contexts
context  = np.array(walkingSettings).reshape(1,3)
contexts = np.repeat(context, numTGSteps, axis=0)

angInit   = bout['angles'][leg][startIdx]
drvInit   = bout['derivatives'][leg][startIdx]
phaseInit = bout['phases'][leg][startIdx]

################################################################################
# TG plus controller and dynamics
################################################################################
TG     = TrajectoryGenerator(filename, leg, numTGSteps, groundModel=ground)
dAct   = int(actDelay / Ts * ctrlSpeedRatio)
print(f'Steps of actuation delay: {dAct}')

legPos  = int(leg[-1])
numAng  = TG._numAng

namesTG = [x[2:] for x in TG._angle_names]
CD      = ControlAndDynamics(leg, Ts/ctrlSpeedRatio, dAct, namesTG)

legIdx         = legs.index(leg)
fullAngleNames = [(leg + ang) for ang in namesTG]
dof            = CD._Nur

numSimSteps   = numTGSteps*ctrlSpeedRatio
angleTG2      = np.zeros((numAng, numTGSteps))
drvTG2        = np.zeros((numAng, numTGSteps))
phaseTG2      = np.zeros(numTGSteps)
angleTG2[:numAng,0], drvTG2[:numAng,0], phaseTG2[0] = angInit, drvInit, phaseInit

xs    = np.zeros([CD._Nx, numSimSteps])
us    = np.zeros([CD._Nu, numSimSteps])

lookahead = math.ceil(dAct/ctrlSpeedRatio)
positions = np.zeros([5, 3, numSimSteps])

# No disturbances for now
dist = np.zeros(CD._Nxr)

for t in range(numSimSteps-1):
    k  = int(t / ctrlSpeedRatio)     # Index for TG data
    kn = int((t+1) / ctrlSpeedRatio) # Next index for TG data
    
    if not (k % ctrlCommRatio) and k != kn and k < numTGSteps-1:
        ang = angleTG2[:numAng,k] + ctrl_to_tg(xs[0:dof,t], legPos, namesTG)
        drv = drvTG2[:numAng,k] + ctrl_to_tg(xs[dof:dof*2,t]*CD._Ts, legPos, namesTG)

        kEnd = min(k+ctrlCommRatio+lookahead, numTGSteps-1)
        angleTG2[:numAng,k+1:kEnd+1], drvTG2[:numAng,k+1:kEnd+1], phaseTG2[k+1:kEnd+1] = \
            TG.get_future_traj(k, kEnd, ang, drv, phaseTG2[k], contexts)

    k1 = min(int((t+dAct) / ctrlSpeedRatio), numTGSteps-1)
    k2 = min(int((t+dAct+1) / ctrlSpeedRatio), numTGSteps-1)
    
    anglesAhead = np.concatenate((angleTG2[:,k1].reshape(numAng,1),
                                  angleTG2[:,k2].reshape(numAng,1)), axis=1)
    drvsAhead   = np.concatenate((drvTG2[:,k1].reshape(numAng,1),
                                  drvTG2[:,k2].reshape(numAng,1)), axis=1)/ctrlSpeedRatio
        
    n1  = CD._Nxr*(dAct+1) # number rows before delayed actuation states
    xf  = xs[n1-CD._Nxr:n1, t]
    ang = angleTG2[:numAng, k2] + ctrl_to_tg(xf[0:numAng], legPos, namesTG)

    # Use ground model on future predicted trajectory, to check if it hits ground
    # Slightly hacky: don't use ground model velocity output
    angNew, junk, groundLegs = ground.step_forward({leg: ang}, {leg: ang}, {leg: ang})

    gndAdjust = 0
    if leg in groundLegs: # Future is predicted to hit ground; account for this
        gndAdjust = tg_to_ctrl(angNew[leg] - ang, legPos, namesTG)
        gndAdjust = np.concatenate((gndAdjust, np.zeros(numAng)))    

    # Propagate dynamics
    us[:,t], xs[:,t+1] = CD.step_forward(xs[:,t], anglesAhead, drvsAhead, dist, gndAdjust)

    # Apply ground interaction to dynamics
    # Slightly hacky: don't use ground model velocity output
    ang = angleTG2[:numAng,kn] + ctrl_to_tg(xs[0:dof,t+1], legPos, namesTG)
    angNew, junk, groundLegs = ground.step_forward({leg: ang}, {leg: ang}, {leg: ang})
    
    if leg in groundLegs:
        # Treat the ground interaction as a disturbance
        angNxt                = tg_to_ctrl(angNew[leg] - angleTG2[:numAng,kn], legPos, namesTG)
        groundDist            = np.zeros(numAng*2)
        groundDist[0:dof]     = angNxt - xs[0:dof,t+1]
        augDist               = CD.get_augmented_dist(groundDist)
        xs[:,t+1]             += augDist


################################################################################
# Postprocessing and plotting
################################################################################
downSamp = list(range(0, numSimSteps, ctrlSpeedRatio))
angle2   = angleTG2 + ctrl_to_tg(xs[0:dof,downSamp], legPos, namesTG)
drv2     = drvTG2 + ctrl_to_tg(xs[dof:dof*2,downSamp]*CD._Ts, legPos, namesTG)

heightsTG2  = np.zeros(numTGSteps)
heights2    = np.zeros(numTGSteps)

gContactTG2  = np.array([None] * numTGSteps)
gContact2    = np.array([None] * numTGSteps)

for t in range(numTGSteps):
    heightsTG2[t]  = ground.get_positions[leg](angleTG2[:,t])[-1, 2]
    heights2[t]    = ground.get_positions[leg](angle2[:,t])[-1, 2]
 
    if heightsTG2[t] <= gndHeight:
        gContactTG2[t] = heightsTG2[t]
    if heights2[t] <= gndHeight:
        gContact2[t] = heights2[t]

mapIx = [namesTG.index(n) for n in anglesCtrl[legPos]]
time  = np.array(range(numTGSteps))

plt.clf()
plt.figure(1)

for i in range(dof):
    plt.subplot(3,dof,i+1)
    plt.title(anglesCtrl[legPos][i])
    idx = mapIx[i]
        
    plt.plot(time, angleTG2[idx,:], 'r', label=f'2Layer TG')    
    plt.plot(time, angle2[idx,:], 'k--', label=f'2Layer')    

    if i==0:
        plt.legend()

    plt.subplot(3,dof,i+dof+1)
    plt.title('Velocity')
    plt.plot(time, drvTG2[idx,:], 'r')    
    plt.plot(time, drv2[idx,:], 'k--')    
            
    plt.subplot(3,1,3)
    plt.title('Height')
    plt.plot(time, heightsTG2, 'r')    
    plt.plot(time, heights2, 'k--')    

    plt.plot(time, gContactTG2, 'r*', markersize=6)    
    plt.plot(time, gContact2, 'k*', markersize=6)    

plt.draw()
plt.show()

