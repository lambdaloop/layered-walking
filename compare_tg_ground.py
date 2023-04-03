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
# filename = '/home/lisa/Downloads/walk_sls_legs_subang_1.pickle'
filename = '/home/lili/data/tuthill/models/models_sls/walk_sls_legs_subang_1.pickle'

walkingSettings = [10, 0, 0] # walking, turning, flipping speeds (mm/s)

numTGSteps      = 200   # How many timesteps to run TG for
Ts              = 1/300 # How fast TG runs
ctrlSpeedRatio  = 2     # Controller will run at Ts / ctrlSpeedRatio
ctrlCommRatio   = 4     # Controller communicates to TG this often (as multiple of Ts)
actDelay        = 0.01  # Seconds; typically 0.02-0.04
senseDelay      = 0.00  # Seconds; typically 0.01

leg     = 'R1'
slipVel = 0


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
# Trajectory generator + ctrl and dynamics, w/o disturbances
################################################################################
dAct   = int(actDelay / Ts * ctrlSpeedRatio)
dSense = int(senseDelay / Ts * ctrlSpeedRatio)
print(f'Steps of actuation delay: {dAct}')
print(f'Steps of sensory delay  : {dSense}')

legPos  = int(leg[-1])

# ground = GroundModel(height=0.75)
ground = GroundModel(offset=[0, 0, -0.8], phi=0, theta=0)

TG      = TrajectoryGenerator(filename, leg, numTGSteps, groundModel=ground)
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
dist  = np.zeros(CD._Nxr) # Zero disturbances

lookahead = math.ceil(dAct/ctrlSpeedRatio)


positions = np.zeros([5, 3, numSimSteps])
# positions2 = np.zeros([5, 3, numSimSteps])

t = 0
k  = int(t / ctrlSpeedRatio)     # Index for TG data
kn = int((t+1) / ctrlSpeedRatio) # Next index for TG data
kEnd = numTGSteps-1
angleTG2[:numAng,k+1:kEnd+1], drvTG2[:numAng,k+1:kEnd+1], phaseTG2[k+1:kEnd+1] = \
            TG.get_future_traj(k, kEnd, angInit, drvInit, phaseTG2[k], contexts)


heights = np.zeros(numTGSteps)
for t in range(numTGSteps):
    heights[t] = ground.get_positions[leg](angleTG2[:,t])[-1, 2]


plt.figure(1)
plt.clf()
plt.subplot(2, 3, 1)
plt.plot(angleTG2.T)
plt.subplot(2, 3, 2)
plt.plot(drvTG2.T)
plt.subplot(2, 3, 3)
plt.plot(heights)
plt.draw()
plt.show(block=False)



# ground = GroundModel(height=0.75)
ground = GroundModel(offset=[0, 0, -0.7], phi=0, theta=0)

TG      = TrajectoryGenerator(filename, leg, numTGSteps, groundModel=None)
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
dist  = np.zeros(CD._Nxr) # Zero disturbances

lookahead = math.ceil(dAct/ctrlSpeedRatio)


positions = np.zeros([5, 3, numSimSteps])
# positions2 = np.zeros([5, 3, numSimSteps])

t = 0
k  = int(t / ctrlSpeedRatio)     # Index for TG data
kn = int((t+1) / ctrlSpeedRatio) # Next index for TG data
kEnd = numTGSteps-1
angleTG2[:numAng,k+1:kEnd+1], drvTG2[:numAng,k+1:kEnd+1], phaseTG2[k+1:kEnd+1] = \
            TG.get_future_traj(k, kEnd, angInit, drvInit, phaseTG2[k], contexts)


heights = np.zeros(numTGSteps)
for t in range(numTGSteps):
    heights[t] = ground.get_positions[leg](angleTG2[:,t])[-1, 2]


plt.figure(1)
plt.subplot(2, 3, 4)
plt.plot(angleTG2.T)
plt.subplot(2, 3, 5)
plt.plot(drvTG2.T)
plt.subplot(2, 3, 6)
plt.plot(heights)
plt.draw()
plt.show(block=False)
