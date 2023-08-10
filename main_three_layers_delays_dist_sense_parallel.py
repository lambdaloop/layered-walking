#!/usr/bin/env python

# only 1 thread, to help parallelize across data
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress warnings from tensorflow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # compute on cpu, it's actually faster for inference with smaller model

# only 1 thread for tf as well
import tensorflow as tf
tf.config.set_soft_device_placement(True)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

import math
import numpy as np
import sys

from tools.ctrl_tools import ControlAndDynamics
from tools.trajgen_tools import TrajectoryGenerator, WalkingData
from tools.angle_functions import legs, \
                            offsets, alphas, kuramato_deriv, \
                            angles_to_pose_names, make_fly_video, \
                            ctrl_to_tg
from tools.dist_tools import *

from tqdm import tqdm, trange
from collections import defaultdict
import os
import pickle
import gc
import time


if len(sys.argv) > 1:
    outfilename = sys.argv[1]
else:
    outfilename = "delays_stats_subang_v2_sense_poisson"

if len(sys.argv) > 2:
    dist_type = sys.argv[2]
else:
    dist_type = 'poisson'

num_batch = 50

start_index = int(sys.argv[3]) * num_batch


################################################################################
# User-defined parameters
################################################################################
# filename = '/home/lisa/Downloads/walk_sls_legs_11.pickle'
# filename = '/home/pierre/data/tuthill/models/models_sls/walk_sls_legs_13.pickle'
# filename = '/home/lili/data/tuthill/models/models_sls/walk_sls_legs_subang_1.pickle'
filename = '/home/pierre/models_sls/walk_sls_legs_subang_6.pickle'


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
if dist_type == 'gaussian':
    distStart = 300
    distEnd   = 301
else:
    distStart = 300
    distEnd   = 600


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

if dist_type == 'gaussian':
    max_velocities = np.arange(0, 10.1, 1.25)
else:
    max_velocities = np.arange(0, 5.1, 0.625)

if dist_type == 'poisson':
    dist_types = [DistType.POISSON_GAUSSIAN]
elif dist_type == 'gaussian':
    dist_types = [DistType.IMPULSE]
else:
    raise ValueError('invalid dist type: received "{}" but should be one of "poisson" or "gaussian"'.format(
        dist_type
    ))

# act_delays = np.arange(0, 0.065, 0.01)
# sense_delays = np.arange(0, 0.045, 0.005)
sense_delays = np.arange(0, 0.020, 0.001)

actDelay = 0.030
dAct = int(actDelay / Ts * ctrlSpeedRatio)

# # 0 delay for this figure
# senseDelay = 0
# dSense = int(senseDelay / Ts * ctrlSpeedRatio)

TG      = [None for i in range(nLegs)]
namesTG = [None for i in range(nLegs)]
for ln, leg in enumerate(legs):
    TG[ln] = TrajectoryGenerator(filename, leg, numTGSteps)
    namesTG[ln] = [x[2:] for x in TG[ln]._angle_names]


full_conditions = [
    {'context': [f_speed, f_rot, f_side],
     'offset': offset,
     'dist': dd,
     'maxVelocity': vel,
     'senseDelay': delay}
    for delay in sense_delays
    for f_speed in fictrac_speeds
    for f_rot in fictrac_rots
    for f_side in fictrac_sides
    for dd in dist_types
    for vel in max_velocities
    for offset in range(4)
]

print(" processing {} / {} ".format(start_index, len(full_conditions)))

if start_index >= len(full_conditions):
    print("  index past length, exiting")
    exit()

conditions = full_conditions[start_index:start_index+num_batch]
actual_sense_delays = list(set([x['senseDelay'] for x in conditions]))

CD_dict = dict()

for senseDelay in tqdm(actual_sense_delays, ncols=70, desc="making controllers"):
    dSense = int(senseDelay / Ts * ctrlSpeedRatio)
    CD = [None for i in range(nLegs)]
    for ln, leg in enumerate(legs):
        # CD[ln] = ControlAndDynamics(leg, Ts/ctrlSpeedRatio, numDelays, futurePenRatio,
        #                             anglePen, drvPen[leg], inputPen, namesTG[ln])
        try:
            CD[ln] = ControlAndDynamics(leg, Ts/ctrlSpeedRatio, dSense, dAct, namesTG[ln])
        except ValueError:
            print("  ERROR leg={}, senseDelay={}, dSense={}".format(
                repr(leg), senseDelay, dSense))
    CD_dict[dSense] = CD



output = defaultdict(list)
store_start = start_index

os.makedirs("output", exist_ok=True)
outname = os.path.join("output", outfilename + "_{:06d}.pkl")
outpath = outname.format(store_start)

if os.path.exists(outpath):
    print("  already processed, exiting!")
    exit()


for ix_cond, cond in enumerate(tqdm(conditions, ncols=70)):
    context = cond['context']
    offset = cond['offset']
    distType = cond['dist']
    maxVelocity = cond['maxVelocity']

    # actDelay = cond['actDelay']
    senseDelay = cond['senseDelay']

    # dAct   = int(actDelay / Ts * ctrlSpeedRatio)
    dSense = int(senseDelay / Ts * ctrlSpeedRatio)

    lookahead   = math.ceil(dAct/ctrlSpeedRatio)

    CD = CD_dict[dSense]

    contexts = [context for _ in range(numSimSteps)]

    # initialize varibales
    angleTG[:] = 0
    drvTG[:] = 0
    phaseTG[:] = 0

    # ys = [None for i in range(nLegs)]
    # xEsts   = [None for i in range(nLegs)]
    # us = [None for i in range(nLegs)]
    # dists = [None for i in range(nLegs)]

    max_nx = max([C._Nx for C in CD])
    max_nu = max([C._Nu for C in CD])
    max_nur = max([C._Nur for C in CD])

    ys    = np.zeros((len(legs), max_nx, numSimSteps))
    xEsts = np.zeros((len(legs), max_nx, numSimSteps))
    us    = np.zeros((len(legs), max_nu, numSimSteps))
    dists = np.zeros((len(legs), max_nur*2, numSimSteps))

    # For height detection and visualization
    heights        = [None for i in range(nLegs)]
    groundContact  = [None for i in range(nLegs)] # For visualization only

    bout = wd.get_initial_vals(context, offset=offset)
    for ln, leg in enumerate(legs):
        numAng = TG[ln]._numAng
        angleTG[ln,:numAng,0] = bout['angles'][leg][0]
        drvTG[ln,:numAng,0] = bout['derivatives'][leg][0]
        phaseTG[ln,0] = bout['phases'][leg][0]

        # ys[ln]    = np.zeros([CD[ln]._Nx, numSimSteps])
        # xEsts[ln] = np.zeros([CD[ln]._Nx, numSimSteps])
        # us[ln]    = np.zeros([CD[ln]._Nu, numSimSteps])
        # dists[ln] = np.zeros([CD[ln]._Nur*2, numSimSteps])

        heights[ln]       = np.array([None] * numSimSteps)
        groundContact[ln] = np.array([None] * numSimSteps)


    distDict = {'maxVelocity' : maxVelocity,
                'rate': 20 * Ts / ctrlSpeedRatio,
                'distType'    : distType
               }


    # Simulation
    for t in range(numSimSteps-1):
        k  = int(t / ctrlSpeedRatio)     # Index for TG data
        kn = int((t+1) / ctrlSpeedRatio) # Next index for TG data


        # Index for future TG data
        k1 = min(int((t+dAct) / ctrlSpeedRatio), numTGSteps-1)
        k2 = min(int((t+dAct+1) / ctrlSpeedRatio), numTGSteps-1)

        # This is only used if TG is updated
        ws = np.zeros(6)
        px = phaseTG[:,k]
        px_half = px + 0.5*Ts * kuramato_deriv(px, alphas, offsets, ws)
        px = px + Ts * kuramato_deriv(px_half, alphas, offsets, ws)
        phaseTG[:,k] = px

        for ln, leg in enumerate(legs):
            legPos  = int(leg[-1])
            legIdx  = legs.index(leg)
            numAng = TG[ln]._numAng
            nu = CD[ln]._Nu
            nx = CD[ln]._Nx
            nur = CD[ln]._Nur

            ang = angleTG[ln,:numAng,k] + ctrl_to_tg(ys[ln][0:nur,t], legPos, namesTG[ln])
            drv = drvTG[ln,:numAng,k] + ctrl_to_tg(
                ys[ln][nur:nur*2,t]*CD[ln]._Ts, legPos, namesTG[ln])

            # Communicate to trajectory generator and get future trajectory
            if not (k % ctrlCommRatio) and k != kn and k < numTGSteps-1:
                kEnd = min(k+ctrlCommRatio+lookahead, numTGSteps-1)
                angleTG[ln,:numAng,k+1:kEnd+1], drvTG[ln,:numAng,k+1:kEnd+1], phaseTG[ln,k+1:kEnd+1] = \
                    TG[ln].get_future_traj(k, kEnd, ang, drv, phaseTG[ln,k], contexts)

            # Apply disturbance
            dist           = get_zero_dists()[leg]
            if k >= distStart and k < distEnd:
                dist = get_dist(distDict, leg)

            anglesAhead = np.concatenate((angleTG[ln,:,k1].reshape(dofTG,1),
                                          angleTG[ln,:,k2].reshape(dofTG,1)), axis=1)
            drvsAhead   = np.concatenate((drvTG[ln,:,k1].reshape(dofTG,1),
                                          drvTG[ln,:,k2].reshape(dofTG,1)), axis=1)/ctrlSpeedRatio

            angleNxt = angleTG[ln,:,kn]

            # us[ln][:,t], ys[ln][:,t+1] = CD[ln].step_forward(ys[ln][:,t], anglesAhead, drvsAhead, dist)
            us[ln][:nu,t], ys[ln][:nx,t+1], xEsts[ln][:nx,t+1] = \
                CD[ln].step_forward(ys[ln][:nx,t], xEsts[ln][:nx,t], anglesAhead, drvsAhead, angleNxt, dist)
            dists[ln][:nur*2, t] = dist


    # True angles sampled at Ts
    # angle    = np.zeros((nLegs, dofTG, numTGSteps))
    downSamp = list(range(ctrlSpeedRatio-1, numSimSteps, ctrlSpeedRatio))
    angle = []
    names = []

    for ln, leg in enumerate(legs):
        numAng = TG[ln]._numAng
        name = TG[ln]._angle_names
        legPos    = int(leg[-1])
        x = angleTG[ln,:numAng,:] + ctrl_to_tg(ys[ln][0:CD[ln]._Nur,downSamp], legPos, namesTG[ln])
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
    # output['ys'].append(np.copy(ys))
    output['dists'].append(np.copy(dists))
    output['angle'].append(np.copy(angs_sim))
    output['angleNames'].append(np.copy(angNames))
    output['conditions'].append(cond)
    output['pose_3d'].append(np.copy(pose_3d))

    # if (ix_cond + 1) % 100 == 0:
    #     outpath = outname.format(store_start)
    #     # np.savez_compressed(outpath, **output)
    #     with open(outpath, 'wb') as f:
    #         pickle.dump(output, f)
    #     del output
    #     gc.collect()
    #     output = defaultdict(list)
    #     store_start = ix_cond + 1

# outpath = outname.format(store_start)
# np.savez_compressed(outpath, **output)
with open(outpath, 'wb') as f:
    pickle.dump(output, f)
