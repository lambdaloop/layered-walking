#!/usr/bin/env python

import math
import matplotlib
import numpy as np
import sys

from tools.ctrl_tools import ControlAndDynamics
from tools.trajgen_tools import TrajectoryGenerator, WalkingData
from tools.angle_functions import legs, anglesTG, anglesCtrl, mapTG2Ctrl, \
                            ctrl_to_tg, tg_to_ctrl, \
                            offsets, alphas, kuramato_deriv, \
                            angles_to_pose_names, make_fly_video
from tools.dist_tools import *

# python3 main_three_layer.py [optional: output file name]
outfilename = 'vids/multileg_3layer.mp4' # default
if len(sys.argv) > 1:
    outfilename = sys.argv[1]

################################################################################
# User-defined parameters
################################################################################
filename = '/home/lisa/Downloads/walk_sls_legs_11.pickle'
#filename = '/home/pierre/data/tuthill/models/models_sls/walk_sls_legs_13.pickle'

walkingSettings = [12, 0, 0] # walking, turning, flipping speeds (mm/s)

numTGSteps     = 550   # How many timesteps to run TG for
Ts             = 1/300 # How fast TG runs
ctrlSpeedRatio = 2     # Controller will run at Ts / ctrlSpeedRatio
ctrlCommRatio  = 8     # Controller communicates to TG this often (as multiple of Ts)
actDelay       = 0.03  # Seconds; typically 0.02-0.04

# LQR penalties
drvPen = {'L1': 1e-5, #
          'L2': 1e-5, #
          'L3': 1e-5, # 
          'R1': 1e-5, # 
          'R2': 1e-5, # 
          'R3': 1e-5  #
         }

futurePenRatio = 1.0 # y_hat(t+1) is penalized (ratio)*pen as much as y(t)
                     # y_hat(t+2) is penalized (ratio^2)*pen as much as y(t)
anglePen       = 1e0
inputPen       = 1e-8

################################################################################
# Disturbance
################################################################################
#distType  = DistType.ZERO
# Will only be applied between t=200 and t=400

#distType = DistType.SLIPPERY_SURFACE
#distDict = {'maxVelocity' : 6}

distType = DistType.UNEVEN_SURFACE
distDict = {'maxHt' : 0.02 * 1e-3}

distDict['distType'] = distType

# Local minima detection parameters (for applying disturbance)
locMinWindow      = 2*ctrlSpeedRatio
nonRepeatWindow   = 10*ctrlSpeedRatio # Assumed minimum distance between minima

################################################################################
# Get walking data
################################################################################
wd       = WalkingData(filename)
bout     = wd.get_bout(walkingSettings)
contexts = bout['contexts']

angInit   = bout['angles']
drvInit   = bout['derivatives']
phaseInit = bout['phases']

################################################################################
# Phase coordinator + trajectory generator + ctrl and dynamics
################################################################################
numDelays   = int(actDelay / Ts * ctrlSpeedRatio)
numSimSteps = numTGSteps*ctrlSpeedRatio
lookahead   = math.ceil(numDelays/ctrlSpeedRatio)

nLegs   = len(legs)
dofTG   = len(anglesTG)
TG      = [None for i in range(nLegs)]
CD      = [None for i in range(nLegs)]

angleTG = np.zeros((nLegs, dofTG, numTGSteps))
drvTG   = np.zeros((nLegs, dofTG, numTGSteps))
phaseTG = np.zeros((nLegs, numTGSteps))

ys      = [None for i in range(nLegs)]
us      = [None for i in range(nLegs)]

# For height detection and visualization
heights        = [None for i in range(nLegs)]
groundContact  = [None for i in range(nLegs)] # For visualization only
lastDetection  = [-nonRepeatWindow for i in range(nLegs)]
fullAngleNames = []

for ln, leg in enumerate(legs):
    fullAngleNames.append([(leg + ang) for ang in anglesTG])

    TG[ln] = TrajectoryGenerator(filename, leg, dofTG, numTGSteps)
    CD[ln] = ControlAndDynamics(leg, Ts/ctrlSpeedRatio, numDelays, futurePenRatio,
                                anglePen, drvPen[leg], inputPen)

    angleTG[ln,:,0], drvTG[ln,:,0], phaseTG[ln,0] = \
        angInit[leg][0], drvInit[leg][0], phaseInit[leg][0]
        
    ys[ln] = np.zeros([CD[ln]._Nx, numSimSteps])
    us[ln] = np.zeros([CD[ln]._Nu, numSimSteps])
    
    heights[ln]       = np.array([None] * numSimSteps)
    groundContact[ln] = np.array([None] * numSimSteps)


# Simulation
for t in range(numSimSteps-1):
    k  = int(t / ctrlSpeedRatio)     # Index for TG data
    kn = int((t+1) / ctrlSpeedRatio) # Next index for TG data
    
    # Index for future TG data
    k1 = min(int((t+numDelays) / ctrlSpeedRatio), numTGSteps-1)
    k2 = min(int((t+numDelays+1) / ctrlSpeedRatio), numTGSteps-1)

    # This is only used if TG is updated
    ws = np.zeros(6)
    px = phaseTG[:,k]
    px_half = px + 0.5*Ts * kuramato_deriv(px, alphas, offsets, ws)*8
    px = px + Ts * kuramato_deriv(px_half, alphas, offsets, ws)*8
    phaseTG[:,k] = px

    for ln, leg in enumerate(legs):
        legPos  = int(leg[-1])
        legIdx  = legs.index(leg)

        ang = angleTG[ln,:,k] + ctrl_to_tg(ys[ln][0:CD[ln]._Nu,t], legPos)
        drv = drvTG[ln,:,k] + ctrl_to_tg(ys[ln][CD[ln]._Nu:,t]*CD[ln]._Ts, legPos)
        
        # Communicate to trajectory generator and get future trajectory
        if not (k % ctrlCommRatio) and k != kn and k < numTGSteps-1:        
            kEnd = min(k+ctrlCommRatio+lookahead, numTGSteps-1)
            angleTG[ln,:,k+1:kEnd+1], drvTG[ln,:,k+1:kEnd+1], phaseTG[ln,k+1:kEnd+1] = \
                TG[ln].get_future_traj(k, kEnd, ang, drv, phaseTG[ln,k], contexts)
        
        # Apply disturbance if in contact with ground
        dist           = get_zero_dists()[leg]    
        heights[ln][t] = get_current_height(ang, fullAngleNames[ln], legIdx)
        if k > 200 and k <= 400 and \
           loc_min_detected(locMinWindow, nonRepeatWindow, lastDetection[ln], heights[ln], t):
            groundContact[ln][t] = heights[ln][t] # Visualize height minimum detection
            lastDetection[ln]    = t
            dist                 = get_dist(distDict, leg)               
        
        anglesAhead = np.concatenate((angleTG[ln,:,k1].reshape(dofTG,1),
                                      angleTG[ln,:,k2].reshape(dofTG,1)), axis=1)
        drvsAhead   = np.concatenate((drvTG[ln,:,k1].reshape(dofTG,1),
                                      drvTG[ln,:,k2].reshape(dofTG,1)), axis=1)/ctrlSpeedRatio
            
        us[ln][:,t], ys[ln][:,t+1] = CD[ln].step_forward(ys[ln][:,t], anglesAhead, drvsAhead, dist)
        
# True angles sampled at Ts
angle    = np.zeros((nLegs, dofTG, numTGSteps))
downSamp = list(range(ctrlSpeedRatio-1, numSimSteps, ctrlSpeedRatio))

for ln, leg in enumerate(legs):
    legPos    = int(leg[-1])
    angle[ln,:,:] = angleTG[ln,:,:] + ctrl_to_tg(ys[ln][0:CD[ln]._Nu,downSamp], legPos)
    
matplotlib.use('Agg')
angs           = angle.reshape(-1, angle.shape[-1]).T
angNames       = [(leg + ang) for leg in legs for ang in anglesTG]
pose_3d        = angles_to_pose_names(angs, angNames)
make_fly_video(pose_3d, outfilename)
