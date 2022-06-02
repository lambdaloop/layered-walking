#!/usr/bin/env python

import matplotlib
import numpy as np

from ctrl_tools import ControlAndDynamics
from trajgen_tools import TrajectoryGenerator
from angle_functions import legs, anglesTG, anglesCtrl, mapTG2Ctrl, \
                            ctrl_to_tg, tg_to_ctrl, \
                            offsets, alphas, kuramato_deriv, \
                            angles_to_pose_names, make_fly_video                            
                            
################################################################################
# User-defined parameters
################################################################################
numSimSteps = 200 # How many timesteps to run model for
Ts          = 1/300 # Sampling time
TGInterval  = 1     # Give feedback to TG once per interval 

# LQR penalties
drvPen = {'L1': 1e1,
          'L2': 1e0,
          'L3': 1e0,
          'R1': 1e1,
          'R2': 1e2,
          'R3': 1e0}
anglePen = 1e0
inputPen = 1e-8

################################################################################
# Phase coordinator + trajectory generator + ctrl and dynamics
################################################################################
filename = '/home/lisa/Downloads/walk_sls_legs_8.pickle'
#filename = '/home/pierre/data/tuthill/models/models_sls/walk_sls_legs_8.pickle'

nLegs   = len(legs)
dofTG   = len(anglesTG)
TG      = [None for i in range(nLegs)]
CD      = [None for i in range(nLegs)]

angleTG = np.zeros((nLegs, dofTG, numSimSteps))
drvTG   = np.zeros((nLegs, dofTG, numSimSteps))
phaseTG = np.zeros((nLegs, numSimSteps))

ys = [None for i in range(nLegs)]
us = [None for i in range(nLegs)]

ang   = np.zeros((nLegs, dofTG))
drv   = np.zeros((nLegs, dofTG))
phase = np.zeros(nLegs)

for ln, leg in enumerate(legs):
    TG[ln] = TrajectoryGenerator(filename, leg, dofTG, numSimSteps)
    CD[ln] = ControlAndDynamics(leg, anglePen, drvPen[leg], inputPen, Ts)
    
    ang[ln], drv[ln], phase[ln] = TG[ln].get_initial_vals()
    angleTG[ln,:,0], drvTG[ln,:,0], phaseTG[ln,0] = ang[ln], drv[ln], phase[ln]
        
    ys[ln] = np.zeros([CD[ln]._Nx, numSimSteps])
    us[ln] = np.zeros([CD[ln]._Nu, numSimSteps])

# Simulation
for t in range(numSimSteps-1):
    ws = np.zeros(6)
    px = phaseTG[:, t]
    px_half = px + 0.5*Ts * kuramato_deriv(px, alphas, offsets, ws)
    px = px + Ts * kuramato_deriv(px_half, alphas, offsets, ws)

    for ln, leg in enumerate(legs):
        legPos   = int(leg[-1])
        
        angleTG[ln, :,t+1], drvTG[ln, :,t+1], phaseTG[ln, t+1] = \
            TG[ln].step_forward(ang[ln], drv[ln], px[ln], TG[ln]._context[t])
        
        us[ln][:,t], ys[ln][:,t+1] = \
            CD[ln].step_forward(ys[ln][:,t], angleTG[ln,:,t], angleTG[ln,:,t+1],
                                drvTG[ln,:,t], drvTG[ln,:,t+1])
        
        ang[ln] = angleTG[ln,:,t+1]
        drv[ln] = drvTG[ln,:,t+1]
        
        if not ((t+1) % TGInterval):
            ang[ln] += ctrl_to_tg(ys[ln][0:CD[ln]._Nu,t+1], legPos)    
            drv[ln] += ctrl_to_tg(ys[ln][CD[ln]._Nu:,t+1]*Ts, legPos)

angle = np.zeros((nLegs, dofTG, numSimSteps))
for ln, leg in enumerate(legs):
    legPos    = int(leg[-1])
    angle[ln] = angleTG[ln] + ctrl_to_tg(ys[ln][0:CD[ln]._Nu,:], legPos)

matplotlib.use('Agg')
angs           = angle.reshape(-1, angle.shape[-1]).T
angNames       = [(leg + ang) for leg in legs for ang in anglesTG]
pose_3d        = angles_to_pose_names(angs, angNames)
make_fly_video(pose_3d, 'vids/multileg_3layer.mp4')

