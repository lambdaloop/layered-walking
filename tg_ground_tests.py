#!/usr/bin/env python

import math
import matplotlib.pyplot as plt
import numpy as np
import sys

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

walkingSettings = [5, 0, 0] # walking, turning, flipping speeds (mm/s)

numTGSteps      = 200   # How many timesteps to run TG for
Ts              = 1/300 # How fast TG runs

leg     = 'L2'

legPos  = int(leg[-1])

################################################################################
# Get walking data
################################################################################
gndHeight = -0.75
offset   = 1
startIdx = 39

wd       = WalkingData(filename)
bout     = wd.get_bout(walkingSettings, offset=offset)

# Use constant contexts
context  = np.array(walkingSettings).reshape(1,3)
contexts = np.repeat(context, numTGSteps, axis=0)

angInit   = bout['angles'][leg][startIdx]
drvInit   = bout['derivatives'][leg][startIdx]
phaseInit = bout['phases'][leg][startIdx]

################################################################################
# Ground model
################################################################################
# Assuming flat ground
ground    = GroundModel(offset=[0, 0, gndHeight], phi=0, theta=0)

################################################################################
# Trajectory generator alone
################################################################################
TGNoGround = TrajectoryGenerator(filename, leg, numTGSteps, groundModel=None)
TG         = TrajectoryGenerator(filename, leg, numTGSteps, groundModel=ground)

namesTG = [x[2:] for x in TG._angle_names]
dof     = len(namesTG)

angleTGng, drvTGng, phaseTGng = TGNoGround.get_future_traj(0, numTGSteps, 
                                    angInit, drvInit, phaseInit, contexts)

angleTG, drvTG, phaseTG = TG.get_future_traj(0, numTGSteps, 
                              angInit, drvInit, phaseInit, contexts)

################################################################################
# Postprocessing and plotting
################################################################################
heightsTGng = np.zeros(numTGSteps)
heightsTG   = np.zeros(numTGSteps)

gContactTGng = np.array([None] * numTGSteps)
gContactTG   = np.array([None] * numTGSteps)

for t in range(numTGSteps):
    heightsTGng[t] = ground.get_positions[leg](angleTGng[:,t])[-1, 2]
    heightsTG[t]   = ground.get_positions[leg](angleTG[:,t])[-1, 2]
 
    if heightsTGng[t] <= gndHeight:
        gContactTGng[t] = heightsTGng[t]
    if heightsTG[t] <= gndHeight:
        gContactTG[t] = heightsTG[t]

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

    if i==0:
        plt.legend()

    plt.subplot(3,dof,i+dof+1)
    plt.title('Velocity')
    plt.plot(time, drvTGng[idx,:], 'b')
    plt.plot(time, drvTG[idx,:], 'g')
            
    plt.subplot(3,1,3)
    plt.title('Height')
    plt.plot(time, heightsTGng, 'b')
    plt.plot(time, heightsTG, 'g')

    plt.plot(time, gContactTGng, 'b*', markersize=6)
    plt.plot(time, gContactTG, 'g*', markersize=6)

plt.draw()
plt.show()

