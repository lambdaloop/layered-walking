#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys

from tools.trajgen_tools import TrajectoryGenerator
from tools.angle_functions import anglesTG, angles_to_pose_names, legs
from tools.dist_tools import get_ground_contact_threshold, get_current_height

# Usage: python3 sandbox_ground_contact.py <leg>
################################################################################
# User-defined parameters
################################################################################
filename = '/home/lisa/Downloads/walk_sls_legs_11.pickle'
#filename = '/home/pierre/data/tuthill/models/models_sls/walk_sls_legs_11.pickle'

numTGSteps  = 200   # How many timesteps to run TG for
Ts          = 1/300 # Sampling time

leg = sys.argv[1]

################################################################################
# Trajectory generator
################################################################################
legPos  = int(leg[-1])
dofTG   = len(anglesTG)
TG      = TrajectoryGenerator(filename, leg, dofTG, numTGSteps)

angleTG = np.zeros((dofTG, numTGSteps))
drvTG   = np.zeros((dofTG, numTGSteps))
phaseTG = np.zeros(numTGSteps)

angleTG[:,0], drvTG[:,0], phaseTG[0] = TG.get_initial_vals()

for t in range(numTGSteps-1):
    angleTG[:,t+1], drvTG[:,t+1], phaseTG[t+1] = \
        TG.step_forward(angleTG[:,t], drvTG[:,t], phaseTG[t], TG._context[t])

################################################################################
# Ground contact detection
################################################################################
legIdx = legs.index(leg)

fullAngleNames = [(leg + ang) for ang in anglesTG]
threshold      = get_ground_contact_threshold(angleTG, fullAngleNames, legIdx)
print(f'Ground Contact Threshold for {leg}: {threshold}')

# Visualize when "ground detection" has occurred, using two methods
groundContact1 = [None] * numTGSteps # Use live detection of local minima (this will be 1 step late)
groundContact2 = [None] * numTGSteps # Use thresholding

heights = []
for t in range(numTGSteps):
    heights.append(get_current_height(angleTG[:,t], fullAngleNames, legIdx))
    if t > 1: # Live detection
        if heights[t-1] < min(heights[t], heights[t-2]): # Minimum occured at t-1
            groundContact1[t] = heights[t]
    if heights[t] < threshold: # Thresholding
        groundContact2[t] = heights[t]

time = np.array(range(numTGSteps))
dof  = len(angleTG) 
plt.figure(1)
plt.clf()
for i in range(dof):
    plt.subplot(3,dof,i+1)
    plt.title(anglesTG[i])
    
    plt.plot(time, angleTG[i,:], 'b')
    
    plt.subplot(3,dof,i+dof+1)
    plt.plot(time, drvTG[i,:], 'b')
    
    plt.subplot(3,dof,2*dof+1)
    plt.plot(time, heights, 'b')
    plt.title('Live detection')
    plt.plot(time, groundContact1, 'r*')        

    plt.subplot(3,dof,2*dof+2)
    plt.plot(time, heights, 'b')
    plt.title('Thresholding')
    plt.plot(time, groundContact2, 'm*')        
    
    plt.plot(time, threshold*np.ones(numTGSteps), 'k-')
plt.show()

