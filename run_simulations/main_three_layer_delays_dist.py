#!/usr/bin/env python

import math
import numpy as np
import sys

from tools.ctrl_tools import ControlAndDynamics
from tools.trajgen_tools import TrajectoryGenerator, WalkingData
from tools.angle_functions import legs, \
                            offsets, alphas, kuramato_deriv, \
                            angles_to_pose_names, make_fly_video
from tools.dist_tools import *

from tqdm import tqdm, trange
from collections import defaultdict
import os

# python3 main_three_layer.py [optional: output file name]
# outfilename = 'vids/multileg_3layer.mp4' # default
# if len(sys.argv) > 1:
    # outfilename = sys.argv[1]

# basename = 'dist_12mms_uneven'
# basename = 'compare_8mms'
# basename = 'dist_12mms_slippery_delay_90ms'

################################################################################
# User-defined parameters
################################################################################
# filename = '/home/lisa/Downloads/walk_sls_legs_11.pickle'
# filename = '/home/pierre/data/tuthill/models/models_sls/walk_sls_legs_13.pickle'
filename = '/home/lili/data/tuthill/models/models_sls/walk_sls_legs_subang_1.pickle'


numTGSteps     = 900   # How many timesteps to run TG for
Ts             = 1/300 # How fast TG runs
ctrlSpeedRatio = 2     # Controller will run at Ts / ctrlSpeedRatio
ctrlCommRatio  = 8     # Controller communicates to TG this often (as multiple of Ts)

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

# disturbance start and end
distStart = 300
distEnd   = 600

# Local minima detection parameters (for applying disturbance)
locMinWindow      = 2*ctrlSpeedRatio
nonRepeatWindow   = 10*ctrlSpeedRatio # Assumed minimum distance between minima

numSimSteps = numTGSteps*ctrlSpeedRatio


wd       = WalkingData(filename)

nLegs   = len(legs)
dofTG   = 5

angleTG = np.zeros((nLegs, dofTG, numTGSteps))
drvTG   = np.zeros((nLegs, dofTG, numTGSteps))
phaseTG = np.zeros((nLegs, numTGSteps))



fictrac_speeds = [6, 8, 10, 12, 14, 16, 18]
fictrac_rots = [0]
fictrac_sides = [0]
# max_velocities = np.arange(0, 25, 5)
max_velocities = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]
dist_types = [DistType.POISSON_GAUSSIAN]
act_delays = np.arange(0, 0.065, 0.015)
# sense_delays = np.arange(0, 0.065, 0.01)

# 0 delay for this figure
senseDelay = 0
dSense = int(senseDelay / Ts * ctrlSpeedRatio)

TG      = [None for i in range(nLegs)]
namesTG = [None for i in range(nLegs)]
for ln, leg in enumerate(legs):
    TG[ln] = TrajectoryGenerator(filename, leg, numTGSteps)
    namesTG[ln] = [x[2:] for x in TG[ln]._angle_names]

CD_dict = dict()


for actDelay in tqdm(act_delays, ncols=70, desc="making controllers"):
    dAct = int(actDelay / Ts * ctrlSpeedRatio)
    CD = [None for i in range(nLegs)]
    for ln, leg in enumerate(legs):
        # CD[ln] = ControlAndDynamics(leg, Ts/ctrlSpeedRatio, numDelays, futurePenRatio,
        #                             anglePen, drvPen[leg], inputPen, namesTG[ln])
        try:
            CD[ln] = ControlAndDynamics(leg, Ts/ctrlSpeedRatio, dSense, dAct, namesTG[ln])
        except ValueError:
            print("  ERROR leg={}, actDelay={}, dAct={}".format(repr(leg), actDelay, dAct)
    CD_dict[dAct] = CD

output = defaultdict(list)

conditions = [
    {'context': [f_speed, f_rot, f_side],
     'offset': offset,
     'dist': dd,
     'maxVelocity': vel,
     'actDelay': delay}
    for f_speed in fictrac_speeds
    for f_rot in fictrac_rots
    for f_side in fictrac_sides
    for dd in dist_types
    for vel in max_velocities
    for delay in act_delays
    for offset in range(4)
]

for cond in tqdm(conditions, ncols=70):
    context = cond['context']
    offset = cond['offset']
    distType = cond['dist']
    maxVelocity = cond['maxVelocity']

    actDelay = cond['actDelay']
    senseDelay = cond['senseDelay']

    dAct   = int(actDelay / Ts * ctrlSpeedRatio)
    dSense = int(senseDelay / Ts * ctrlSpeedRatio)

    lookahead   = math.ceil(numDelays/ctrlSpeedRatio)

    CD = CD_dict[dAct]

    contexts = [context for _ in range(numSimSteps)]

    # initialize varibales
    angleTG[:] = 0
    drvTG[:] = 0
    phaseTG[:] = 0

    ys = [None for i in range(nLegs)]
    us = [None for i in range(nLegs)]
    dists = [None for i in range(nLegs)]

    # For height detection and visualization
    heights        = [None for i in range(nLegs)]
    groundContact  = [None for i in range(nLegs)] # For visualization only
    lastDetection  = [-nonRepeatWindow for i in range(nLegs)]

    bout = wd.get_initial_vals(context, offset=offset)
    for ln, leg in enumerate(legs):
        numAng = TG[ln]._numAng
        angleTG[ln,:numAng,0] = bout['angles'][leg][0]
        drvTG[ln,:numAng,0] = bout['derivatives'][leg][0]
        phaseTG[ln,0] = bout['phases'][leg][0]

        ys[ln]    = np.zeros([CD[ln]._Nx, numSimSteps])
        us[ln]    = np.zeros([CD[ln]._Nu, numSimSteps])
        dists[ln] = np.zeros([CD[ln]._Nu*2, numSimSteps])

        heights[ln]       = np.array([None] * numSimSteps)
        groundContact[ln] = np.array([None] * numSimSteps)

    lastDetection  = [-nonRepeatWindow for i in range(nLegs)]

    distDict = {'maxVelocity' : maxVelocity,         # Slippery surface
                # 'maxHt'       : 0.0015 * 1e-3,   # Uneven surface
                # 'height'      : -0.1/1000,  # Stepping on a bump/pit
                # 'distLeg'     : 'L1',       # Stepping on a bump/pit
                # 'angle'       : 10,         # Walking on slope (degrees)
                # 'missingLeg'  : 'L1',        # Missing leg
                'distType'    : distType
               }


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
            numAng = TG[ln]._numAng

            ang = angleTG[ln,:numAng,k] + ctrl_to_tg(ys[ln][0:CD[ln]._Nu,t], legPos, namesTG[ln])
            drv = drvTG[ln,:numAng,k] + ctrl_to_tg(
                ys[ln][CD[ln]._Nu:CD[ln]._Nu*2,t]*CD[ln]._Ts, legPos, namesTG[ln])

            # Communicate to trajectory generator and get future trajectory
            if not (k % ctrlCommRatio) and k != kn and k < numTGSteps-1:
                kEnd = min(k+ctrlCommRatio+lookahead, numTGSteps-1)
                angleTG[ln,:numAng,k+1:kEnd+1], drvTG[ln,:numAng,k+1:kEnd+1], phaseTG[ln,k+1:kEnd+1] = \
                    TG[ln].get_future_traj(k, kEnd, ang, drv, phaseTG[ln,k], contexts)

            # Apply disturbance if in contact with ground
            dist           = get_zero_dists()[leg]
            heights[ln][t] = get_current_height(ang, TG[ln]._angle_names, legIdx)
            if k > distStart and k <= distEnd and \
               loc_min_detected(locMinWindow, nonRepeatWindow, lastDetection[ln], heights[ln], t):
                groundContact[ln][t] = heights[ln][t] # Visualize height minimum detection
                lastDetection[ln]    = t
                dist                 = get_dist(distDict, leg)

            anglesAhead = np.concatenate((angleTG[ln,:,k1].reshape(dofTG,1),
                                          angleTG[ln,:,k2].reshape(dofTG,1)), axis=1)
            drvsAhead   = np.concatenate((drvTG[ln,:,k1].reshape(dofTG,1),
                                          drvTG[ln,:,k2].reshape(dofTG,1)), axis=1)/ctrlSpeedRatio

            us[ln][:,t], ys[ln][:,t+1] = CD[ln].step_forward(ys[ln][:,t], anglesAhead, drvsAhead, dist)
            dists[ln][:, t] = dist



    # True angles sampled at Ts
    # angle    = np.zeros((nLegs, dofTG, numTGSteps))
    downSamp = list(range(ctrlSpeedRatio-1, numSimSteps, ctrlSpeedRatio))
    angle = []
    names = []

    for ln, leg in enumerate(legs):
        numAng = TG[ln]._numAng
        name = TG[ln]._angle_names
        legPos    = int(leg[-1])
        x = angleTG[ln,:numAng,:] + ctrl_to_tg(ys[ln][0:CD[ln]._Nu,downSamp], legPos, namesTG[ln])
        angle.append(x)
        names.append(name)

    # matplotlib.use('Agg')
    # angs           = angle.reshape(-1, angle.shape[-1]).T
    # angNames       = [(leg + ang) for leg in legs for ang in anglesTG]
    angs_sim = np.vstack(angle).T
    angNames = np.hstack(names)
    pose_3d = angles_to_pose_names(angs_sim, angNames)

    output['angleTG'].append(np.copy(angleTG))
    output['drvTG'].append(np.copy(drvTG))
    output['phaseTG'].append(np.copy(phaseTG))
    output['us'].append(np.copy(us))
    output['ys'].append(np.copy(ys))
    output['angle'].append(np.copy(angs_sim))
    output['angleNames'].append(np.copy(angNames))
    output['conditions'].append(cond)
    output['pose_3d'].append(np.copy(pose_3d))

os.makedirs("output", exist_ok=True)
outname = os.path.join("output", "delays_stats_subang_v2.npz")

np.savez_compressed(outname, **output)
