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
senseDelay      = 0.01  # Seconds; typically 0.01

leg = 'R1'

################################################################################
# Get walking data
################################################################################
wd       = WalkingData(filename)
bout     = wd.get_bout(walkingSettings, offset=4)

# Use constant contexts
context  = np.array(walkingSettings).reshape(1,3)
contexts = np.repeat(context, numTGSteps, axis=0)

angInit   = bout['angles'][leg][30]
drvInit   = bout['derivatives'][leg][30]
phaseInit = bout['phases'][leg][30]

################################################################################
# Ground model
################################################################################
# Assuming flat ground
gndHeight = -0.85
ground    = GroundModel(offset=[0, 0, gndHeight], phi=0, theta=0)

################################################################################
# Trajectory generator alone
################################################################################
TGNoGround = TrajectoryGenerator(filename, leg, numTGSteps, groundModel=None)
TG         = TrajectoryGenerator(filename, leg, numTGSteps, groundModel=ground)

angleTGng, drvTGng, phaseTGng = TGNoGround.get_future_traj(0, numTGSteps, 
                                    angInit, drvInit, phaseInit, contexts)

angleTG, drvTG, phaseTG = TG.get_future_traj(0, numTGSteps, 
                              angInit, drvInit, phaseInit, contexts)

################################################################################
# TG plus controller and dynamics
################################################################################
dAct   = int(actDelay / Ts * ctrlSpeedRatio)
dSense = int(senseDelay / Ts * ctrlSpeedRatio)
print(f'Steps of actuation delay: {dAct}')
print(f'Steps of sensory delay  : {dSense}')

legPos  = int(leg[-1])
numAng  = TG._numAng

namesTG = [x[2:] for x in TG._angle_names]
CD      = ControlAndDynamics(leg, Ts/ctrlSpeedRatio, dSense, dAct, namesTG)

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

lookahead = math.ceil(dAct/ctrlSpeedRatio)
positions = np.zeros([5, 3, numSimSteps])

# No disturbances for now
dist = np.zeros(CD._Nxr)

for t in range(numSimSteps-1):
    k  = int(t / ctrlSpeedRatio)     # Index for TG data
    kn = int((t+1) / ctrlSpeedRatio) # Next index for TG data

    ang   = angleTG2[:numAng,k] + ctrl_to_tg(xs[0:dof,t], legPos, namesTG)
    
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
    us[:,t], xs[:,t+1], xEsts[:,t+1] = CD.step_forward(xs[:,t], xEsts[:,t], anglesAhead, drvsAhead, dist)

    # Ground model
    ang_prev = angleTG2[:numAng,k] + ctrl_to_tg(xs[0:dof,t], legPos, namesTG)
    ang      = angleTG2[:numAng,kn] + ctrl_to_tg(xs[0:dof,t+1], legPos, namesTG)
    drv      = drvTG2[:numAng,kn] + ctrl_to_tg(xs[dof:dof*2,t+1]*CD._Ts, legPos, namesTG)

    ang_new_dict, drv_new_dict, ground_legs = ground.step_forward(
        {leg: ang_prev}, {leg: ang}, {leg: drv})
        
    ang_next = ang_new_dict[leg]
    drv_next = drv_new_dict[leg]
    
    if leg in ground_legs:
        angleChange = ang_next - angleTG[:numAng,kn]
        print(f'time: {k}, angle change (deg): {angleChange}')
        xs[0:dof,t+1] = tg_to_ctrl(ang_next - angleTG2[:numAng,kn], legPos, namesTG)   
    
################################################################################
# Postprocessing and plotting
################################################################################
downSamp = list(range(0, numSimSteps, ctrlSpeedRatio))
angle2   = angleTG2 + ctrl_to_tg(xs[0:dof,downSamp], legPos, namesTG)
drv2     = drvTG2 + ctrl_to_tg(xs[dof:dof*2,downSamp]*CD._Ts, legPos, namesTG)

heightsTGng = np.zeros(numTGSteps)
heightsTG   = np.zeros(numTGSteps)
heightsTG2  = np.zeros(numTGSteps)
heights2    = np.zeros(numTGSteps)

gContactTGng = np.array([None] * numTGSteps)
gContactTG   = np.array([None] * numTGSteps)
gContactTG2  = np.array([None] * numTGSteps)
gContact2    = np.array([None] * numTGSteps)

for t in range(numTGSteps):
    heightsTGng[t] = ground.get_positions[leg](angleTGng[:,t])[-1, 2]
    heightsTG[t]   = ground.get_positions[leg](angleTG[:,t])[-1, 2]
    heightsTG2[t]  = ground.get_positions[leg](angleTG2[:,t])[-1, 2]
    heights2[t]    = ground.get_positions[leg](angle2[:,t])[-1, 2]
 
    if heightsTGng[t] <= gndHeight:
        gContactTGng[t] = heightsTGng[t]
    if heightsTG[t] <= gndHeight:
        gContactTG[t] = heightsTG[t]
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
        
    plt.plot(time, angleTGng[idx,:], 'b', label=f'TG No Ground')
    plt.plot(time, angleTG[idx,:], 'g', label=f'TG Ground')    
    plt.plot(time, angleTG2[idx,:], 'r', label=f'2Layer TG')    
    plt.plot(time, angle2[idx,:], 'k--', label=f'2Layer')    
    plt.ylim(-180, 180)

    if i==0:
        plt.legend()

    plt.subplot(3,dof,i+dof+1)
    plt.title('Velocity')
    plt.plot(time, drvTGng[idx,:], 'b')
    plt.plot(time, drvTG[idx,:], 'g')
    plt.plot(time, drvTG2[idx,:], 'r')    
    plt.plot(time, drv2[idx,:], 'k--')    
            
    plt.subplot(3,1,3)
    plt.title('Height')
    plt.plot(time, heightsTGng, 'b')
    plt.plot(time, heightsTG, 'g')
    plt.plot(time, heightsTG2, 'r')    
    plt.plot(time, heights2, 'k--')    

    plt.plot(time, gContactTGng, 'b*', markersize=6)
    plt.plot(time, gContactTG, 'g*', markersize=6)
    plt.plot(time, gContactTG2, 'r*', markersize=6)    
    plt.plot(time, gContact2, 'k*', markersize=6)    

plt.draw()
plt.show()

