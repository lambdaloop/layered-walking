#!/usr/bin/env python

import matplotlib
import numpy as np

from tools.trajgen_tools import TrajectoryGenerator, WalkingData
from tools.angle_functions import legs, anglesTG, offsets, alphas, kuramato_deriv, \
                            angles_to_pose_names, make_fly_video
                            
################################################################################
# User-defined parameters
################################################################################
numSimSteps = 200 # How many timesteps to run model for
Ts          = 1/300 # Sampling time

################################################################################
# Phase coordinator + trajectory generator
################################################################################
filename = '/home/lisa/Downloads/walk_sls_legs_11.pickle'
#filename = '/home/pierre/data/tuthill/models/models_sls/walk_sls_legs_11.pickle'

nLegs   = len(legs)
dofTG   = len(anglesTG)
TG      = [None for i in range(nLegs)]
angleTG = np.zeros((nLegs, dofTG, numSimSteps))
drvTG   = np.zeros((nLegs, dofTG, numSimSteps))
phaseTG = np.zeros((nLegs, numSimSteps))

wd       = WalkingData(filename)
bout     = wd.get_bout([15, 0, 0])
contexts = bout['contexts']

for ln, leg in enumerate(legs):
    TG[ln]          = TrajectoryGenerator(filename, leg, dofTG, numSimSteps)
    angleTG[ln,:,0] = bout['angles'][leg][0]
    drvTG[ln,:,0]   = bout['derivatives'][leg][0]
    phaseTG[ln,0]   = bout['phases'][leg][0]

for t in range(numSimSteps-1):
    ws = np.zeros(6)
    px = phaseTG[:, t]
    px_half = px + 0.5*Ts * kuramato_deriv(px, alphas, offsets, ws)
    px = px + Ts * kuramato_deriv(px_half, alphas, offsets, ws)

    for ln in range(nLegs):
        angleTG[ln, :,t+1], drvTG[ln, :,t+1], phaseTG[ln, t+1] = \
            TG[ln].step_forward(angleTG[ln, :,t], drvTG[ln, :,t],
                                    px[ln], contexts[t])

matplotlib.use('Agg')
angs           = angleTG.reshape(-1, angleTG.shape[-1]).T
angNames       = [(leg + ang) for leg in legs for ang in anglesTG]
pose_3d        = angles_to_pose_names(angs, angNames)
make_fly_video(pose_3d, 'vids/multileg_pc_tg.mp4')

