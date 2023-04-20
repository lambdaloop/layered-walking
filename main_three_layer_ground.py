#!/usr/bin/env python

import math
import matplotlib
import numpy as np
import sys

from tqdm import tqdm, trange

from tools.ctrl_tools import ControlAndDynamics
from tools.trajgen_tools import TrajectoryGenerator, WalkingData
from tools.angle_functions import legs, anglesTG, anglesCtrl, \
                            ctrl_to_tg, tg_to_ctrl, \
                            offsets, alphas, kuramato_deriv, \
                            angles_to_pose_names, make_fly_video
from tools.dist_tools import *
from tools.ground_model import GroundModel


# python3 main_three_layer.py [optional: output file name]
# outfilename = 'vids/multileg_3layer.mp4' # default
# if len(sys.argv) > 1:
    # outfilename = sys.argv[1]

# basename = 'dist_12mms_uneven'
# basename = 'compare_8mms'
# basename = 'dist_12mms_slippery_delay_30ms'
basename = 'test_ground_optim'

################################################################################
# User-defined parameters
################################################################################
# filename = '/home/lisa/Downloads/walk_sls_legs_subang_1.pickle'
# filename = '/home/lili/data/tuthill/models/models_sls/walk_sls_legs_13.pickle'
filename = '/home/lili/data/tuthill/models/models_sls/walk_sls_legs_subang_1.pickle'

walkingSettings = [14, 0, 0] # walking, turning, flipping speeds (mm/s)

numTGSteps     = 120   # How many timesteps to run TG for
Ts             = 1/300 # How fast TG runs
ctrlSpeedRatio = 2     # Controller will run at Ts / ctrlSpeedRatio
ctrlCommRatio  = 8     # Controller communicates to TG this often (as multiple of Ts)
actDelay       = 0.00  # Seconds; typically 0.02-0.04
senseDelay     = 0.00  # Seconds; typically 0.01
couplingDelay  = 0.010



################################################################################
# Disturbance
################################################################################
boutNum  = 4 # Default is 0; change bouts for different random behaviors

distStart = 100
distEnd   = 300

# distType = DistType.SLIPPERY_SURFACE
distType = DistType.ZERO
distDict = {'maxVelocity' : 1}
distDict['distType'] = distType

# Local minima detection parameters (for applying disturbance)
locMinWindow      = 2*ctrlSpeedRatio
nonRepeatWindow   = 10*ctrlSpeedRatio # Assumed minimum distance between minima

################################################################################
# Get walking data
################################################################################
wd       = WalkingData(filename)
bout     = wd.get_bout(walkingSettings, offset=boutNum)
print(bout['angles'])

# Use constant contexts
context  = np.array(walkingSettings).reshape(1,3)
contexts = np.repeat(context, numTGSteps, axis=0)

angInit   = bout['angles']
drvInit   = bout['derivatives']
phaseInit = bout['phases']

################################################################################
# Phase coordinator + trajectory generator + ctrl and dynamics
################################################################################
dAct   = int(actDelay / Ts * ctrlSpeedRatio)
dSense = int(senseDelay / Ts * ctrlSpeedRatio)
print(f'Steps of actuation delay: {dAct}')
print(f'Steps of sensory delay  : {dSense}')

numSimSteps = numTGSteps*ctrlSpeedRatio
lookahead   = math.ceil(dAct/ctrlSpeedRatio)

numDelaysCoupling = int(round(couplingDelay / Ts))

nLegs   = len(legs)
dofTG   = len(anglesTG)
TG      = [None for i in range(nLegs)]
CD      = [None for i in range(nLegs)]
namesTG = [None for i in range(nLegs)]

angleTG = np.zeros((nLegs, dofTG, numTGSteps))
drvTG   = np.zeros((nLegs, dofTG, numTGSteps))
phaseTG = np.zeros((nLegs, numTGSteps))

xs      = [None for i in range(nLegs)]
xEsts   = [None for i in range(nLegs)]
us      = [None for i in range(nLegs)]

# For height detection and visualization
heights        = [None for i in range(nLegs)]
groundContact  = [None for i in range(nLegs)] # For visualization only
lastDetection  = [-nonRepeatWindow for i in range(nLegs)]
fullAngleNames = []

ground = GroundModel(offset=[0, 0, -0.8], phi=-18, theta=0)
# ground = None

# groundTG = ground
groundTG = None

for ln, leg in enumerate(legs):    
    TG[ln] = TrajectoryGenerator(filename, leg, numTGSteps, groundModel=groundTG)

    fullAngleNames.append(TG[ln]._angle_names)

    namesTG[ln] = [x[2:] for x in TG[ln]._angle_names]
    CD[ln] = ControlAndDynamics(leg, Ts/ctrlSpeedRatio, dSense, dAct, namesTG[ln])
    fullAngleNames.append([(leg + ang) for ang in namesTG[ln]])
    numAng = TG[ln]._numAng

    angleTG[ln,:numAng,0], drvTG[ln,:numAng,0], phaseTG[ln,0] = \
        angInit[leg][0], drvInit[leg][0], phaseInit[leg][0]
    
    xs[ln]    = np.zeros([CD[ln]._Nx, numSimSteps])
    xEsts[ln] = np.zeros([CD[ln]._Nx, numSimSteps])
    us[ln]    = np.zeros([CD[ln]._Nu, numSimSteps])
    
    heights[ln]       = np.array([None] * numSimSteps)
    groundContact[ln] = np.array([None] * numSimSteps)



# Simulation
for t in trange(numSimSteps-1, ncols=70):
    k  = int(t / ctrlSpeedRatio)     # Index for TG data
    kn = int((t+1) / ctrlSpeedRatio) # Next index for TG data
    kc = max(k - numDelaysCoupling, 0)
    
    # Index for future TG data
    k1 = min(int((t+dAct) / ctrlSpeedRatio), numTGSteps-1)
    k2 = min(int((t+dAct+1) / ctrlSpeedRatio), numTGSteps-1)

    # This is only used if TG is updated
    ws = np.zeros(6)
    px = phaseTG[:,kc]
    px_half = px + 0.5*Ts * kuramato_deriv(px, alphas, offsets, ws)*8
    phaseTG[:,k] = phaseTG[:,k] + Ts * kuramato_deriv(px_half, alphas, offsets, ws)*8
    # phaseTG[:,k] = pxk

    for ln, leg in enumerate(legs):
        legPos  = int(leg[-1])
        legIdx  = legs.index(leg)
        numAng = TG[ln]._numAng

        ang = angleTG[ln,:numAng,k] + ctrl_to_tg(xs[ln][0:CD[ln]._Nur,t], legPos, namesTG[ln])
        drv = drvTG[ln,:numAng,k] + ctrl_to_tg(
            xs[ln][CD[ln]._Nur:CD[ln]._Nur*2,t]*CD[ln]._Ts, legPos, namesTG[ln])
        
        # Communicate to trajectory generator and get future trajectory
        if not (k % ctrlCommRatio) and k != kn and k < numTGSteps-1:        
            kEnd = min(k+ctrlCommRatio+lookahead, numTGSteps-1)
            angleTG[ln,:numAng,k+1:kEnd+1], drvTG[ln,:numAng,k+1:kEnd+1], phaseTG[ln,k+1:kEnd+1] = \
                TG[ln].get_future_traj(k, kEnd, ang, drv, phaseTG[ln,k], contexts)
        
        # Apply disturbance if in contact with ground
        dist           = get_zero_dists()[leg]
        # heights[ln][t] = get_current_height(ang, fullAngleNames[ln], legIdx)
        # if k > distStart and k <= distEnd and \
        #    loc_min_detected(locMinWindow, nonRepeatWindow, lastDetection[ln], heights[ln], t):
        #     groundContact[ln][t] = heights[ln][t] # Visualize height minimum detection
        #     lastDetection[ln]    = t
        #     dist                 = get_dist(distDict, leg)

        anglesAhead = np.concatenate((angleTG[ln,:,k1].reshape(dofTG,1),
                                      angleTG[ln,:,k2].reshape(dofTG,1)), axis=1)
        drvsAhead   = np.concatenate((drvTG[ln,:,k1].reshape(dofTG,1),
                                      drvTG[ln,:,k2].reshape(dofTG,1)), axis=1)/ctrlSpeedRatio
            
        us[ln][:,t], xs[ln][:,t+1], xEsts[ln][:,t+1] = \
            CD[ln].step_forward(xs[ln][:,t], xEsts[ln][:,t], anglesAhead, drvsAhead, dist)


    ## simulate the ground
    if ground is not None:
        # get the current angles
        ang_prev_dict = dict()
        ang_dict = dict()
        drv_dict = dict()
        for ln, leg in enumerate(legs):
            legPos    = int(leg[-1])
            dofCD = CD[ln]._Nur
            numAng = TG[ln]._numAng
            ang_prev_dict[leg] = angleTG[ln,:numAng,k] + ctrl_to_tg(xs[ln][0:dofCD,t], legPos, namesTG[ln])
            ang_dict[leg] = angleTG[ln,:numAng,kn] + ctrl_to_tg(xs[ln][0:dofCD,t+1], legPos, namesTG[ln])
            drv_dict[leg] = drvTG[ln,:numAng,kn] + ctrl_to_tg(xs[ln][dofCD:dofCD*2,t+1]*CD[ln]._Ts, legPos, namesTG[ln])

        # update the angles
        ang_new_dict, drv_new_dict, ground_legs = ground.step_forward(ang_prev_dict, ang_dict, drv_dict)

        # print(t, ground_legs)

        # update xs
        for ln, leg in enumerate(legs):
            legPos    = int(leg[-1])
            dofCD = CD[ln]._Nur
            numAng = TG[ln]._numAng
            xs[ln][0:dofCD,t+1] = tg_to_ctrl(
                ang_new_dict[leg] - angleTG[ln,:numAng,kn], legPos, namesTG[ln])

# True angles sampled at Ts
# angle    = np.zeros((nLegs, dofTG, numTGSteps))
downSamp = list(range(ctrlSpeedRatio-1, numSimSteps, ctrlSpeedRatio))
angle = []
names = []

for ln, leg in enumerate(legs):
    numAng = TG[ln]._numAng
    name = TG[ln]._angle_names
    legPos    = int(leg[-1])
    x = angleTG[ln,:numAng,:] + ctrl_to_tg(xs[ln][0:CD[ln]._Nur,downSamp], legPos, namesTG[ln])
    angle.append(x)
    names.append(name)


matplotlib.use('Agg')
# angs           = angle.reshape(-1, angle.shape[-1]).T
# angNames       = [(leg + ang) for leg in legs for ang in anglesTG]
angs_sim = np.vstack(angle).T
angNames = np.hstack(names)
pose_3d        = angles_to_pose_names(angs_sim, angNames)
# make_fly_video(pose_3d, outfilename)
# make_fly_video(pose_3d, 'vids/{}_h{:02d}.mp4'.format(basename, int(ground._height*100)))
make_fly_video(pose_3d, 'vids/{}.mp4'.format(basename), ground=ground)

angs_real = np.hstack([bout['angles'][leg] for leg in legs])
p3d = angles_to_pose_names(angs_real, angNames)
# make_fly_video(p3d, 'vids/{}_real.mp4'.format(basename))

wanted_angles = ['L1C_flex', 'L2B_rot', 'L3C_flex', 'R1C_flex', 'R2B_rot', 'R3C_flex']
ixs = []
for name in wanted_angles:
    ix = np.where(angNames == name)[0][0]
    ixs.append(ix)

import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
# ix = np.where(angNames == 'L1C_flex')[0]
plt.figure(1)
plt.clf()
# plt.plot(angs_sim[:, ixs])
# plt.plot(angs_real[:, ix])
plt.plot(pose_3d[:, :, -1, -1])
plt.draw()
plt.show(block=False)

