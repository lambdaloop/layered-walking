#!/usr/bin/env python

import matplotlib
import numpy as np

from tools.ctrl_tools import ControlAndDynamics
from tools.trajgen_tools import TrajectoryGenerator, WalkingData
from tools.angle_functions import legs, anglesTG, anglesCtrl, mapTG2Ctrl, \
                            ctrl_to_tg, tg_to_ctrl, \
                            offsets, alphas, kuramato_deriv, \
                            angles_to_pose_names, make_fly_video                            

from collections import defaultdict
from tqdm import tqdm
import os
################################################################################
# User-defined parameters
################################################################################
# filename = '/home/lisa/Downloads/walk_sls_legs_11.pickle'
filename = '/home/pierre/data/tuthill/models/models_sls/walk_sls_legs_11.pickle'

numTGSteps  = 200 # How many timesteps to run TG for
Ts          = 1/300 # Sampling time
ctrlTsRatio = 5    # Controller will sample at Ts / ctrlTsRatio

# LQR penalties
drvPen = {'L1': 1e-2,
          'L2': 1e-2,
          'L3': 1e-2,
          'R1': 1e-2,
          'R2': 1e-2,
          'R3': 1e-2}
anglePen = 1e0
inputPen = 1e-8

################################################################################
# Phase coordinator + trajectory generator + ctrl and dynamics
################################################################################
numSimSteps = numTGSteps*ctrlTsRatio

nLegs   = len(legs)
dofTG   = len(anglesTG)
TG      = [None for i in range(nLegs)]
CD      = [None for i in range(nLegs)]

angleTG = np.zeros((nLegs, dofTG, numTGSteps))
drvTG   = np.zeros((nLegs, dofTG, numTGSteps))
phaseTG = np.zeros((nLegs, numTGSteps))

ys    = [None for i in range(nLegs)]
us    = [None for i in range(nLegs)]
dists = [None for i in range(nLegs)]

ang   = np.zeros((nLegs, dofTG))
drv   = np.zeros((nLegs, dofTG))
phase = np.zeros(nLegs)

wd = WalkingData(filename)


for ln, leg in enumerate(legs):
    TG[ln] = TrajectoryGenerator(filename, leg, dofTG, numTGSteps)
    CD[ln] = ControlAndDynamics(leg, anglePen, drvPen[leg], inputPen, Ts/ctrlTsRatio, 0)

fictrac_speeds = [4, 8, 12, 16]
fictrac_rots = [-8, -4, 0, 4, 8]
fictrac_sides = [-4, -2, 0, 2, 4]

output = defaultdict(list)

angNames = [(leg + ang) for leg in legs for ang in anglesTG]

conditions = [
    {'context': [f_speed, f_rot, f_side], 'offset': offset}
    for f_speed in fictrac_speeds
    for f_rot in fictrac_rots
    for f_side in fictrac_sides
    for offset in range(5)
]

for cond in tqdm(conditions, ncols=70):
    context = cond['context']
    offset = cond['offset']

    bout = wd.get_initial_vals(context, offset=offset)
    for ln, leg in enumerate(legs):
        ang[ln] = bout['angles'][leg][0]
        drv[ln] = bout['derivatives'][leg][0]
        phase[ln] = bout['phases'][leg][0]

        # ang[ln], drv[ln], phase[ln] = TG[ln].get_initial_vals()
        angleTG[ln,:,0], drvTG[ln,:,0], phaseTG[ln,0] = ang[ln], drv[ln], phase[ln]

        ys[ln]    = np.zeros([CD[ln]._Nx, numSimSteps])
        us[ln]    = np.zeros([CD[ln]._Nu, numSimSteps])
        dists[ln] = np.zeros([CD[ln]._Nx, numSimSteps])

    # context = bout['contexts']

    # Simulation
    for t in range(numSimSteps-1):
        k  = int(t / ctrlTsRatio)      # Index for TG/PC
        kn = int((t+1) / ctrlTsRatio)  # Next index for TG/PC

        # This is only used if TG is updated
        ws = np.zeros(6)
        px = phaseTG[:,k]
        px_half = px + 0.5*Ts * kuramato_deriv(px, alphas, offsets, ws)
        px = px + Ts * kuramato_deriv(px_half, alphas, offsets, ws)

        for ln, leg in enumerate(legs):
            if not ((t+1) % ctrlTsRatio):
                legPos  = int(leg[-1])
                ang[ln] = angleTG[ln,:,k] + ctrl_to_tg(ys[ln][0:CD[ln]._Nu,t], legPos)
                drv[ln] = drvTG[ln,:,k] + ctrl_to_tg(ys[ln][CD[ln]._Nu:,t]*CD[ln]._Ts, legPos)

                angleTG[ln,:,kn], drvTG[ln,:,kn], phaseTG[ln,kn] = \
                    TG[ln].step_forward(ang[ln], drv[ln], px[ln], context)

                us[ln][:,t], ys[ln][:,t+1] = \
                    CD[ln].step_forward(ys[ln][:,t], angleTG[ln,:,k], angleTG[ln,:,kn],
                                        drvTG[ln,:,k]/ctrlTsRatio, drvTG[ln,:,kn]/ctrlTsRatio, dists[ln][:,t])

    # True angles sampled at Ts
    angle    = np.zeros((nLegs, dofTG, numTGSteps))
    downSamp = list(range(ctrlTsRatio-1, numSimSteps, ctrlTsRatio))

    for ln, leg in enumerate(legs):
        legPos    = int(leg[-1])
        angle[ln,:,:] = angleTG[ln,:,:] + ctrl_to_tg(ys[ln][0:CD[ln]._Nu,downSamp], legPos)

    output['angleTG'].append(angleTG)
    output['drvTG'].append(drvTG)
    output['phaseTG'].append(phaseTG)
    output['us'].append(us)
    output['ys'].append(ys)

    output['angle'].append(angle)

    output['conditions'].append(cond)

    angs = angle.reshape(-1, angle.shape[-1]).T
    pose_3d = angles_to_pose_names(angs, angNames)

    output['pose_3d'].append(pose_3d)

os.makedirs("output", exist_ok=True)
outname = os.path.join("output", "control_stats.npz")

np.savez_compressed(outname, **output)
