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
filename = '/home/lisa/Downloads/walk_sls_legs_8.pickle'
#filename = '/home/pierre/data/tuthill/models/models_sls/walk_sls_legs_8.pickle'

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

for ln, leg in enumerate(legs):
    TG[ln] = TrajectoryGenerator(filename, leg, dofTG, numTGSteps)
    CD[ln] = ControlAndDynamics(leg, anglePen, drvPen[leg], inputPen, Ts/ctrlTsRatio)
    
    ang[ln], drv[ln], phase[ln] = TG[ln].get_initial_vals()
    angleTG[ln,:,0], drvTG[ln,:,0], phaseTG[ln,0] = ang[ln], drv[ln], phase[ln]
        
    ys[ln]    = np.zeros([CD[ln]._Nx, numSimSteps])
    us[ln]    = np.zeros([CD[ln]._Nu, numSimSteps])
    dists[ln] = np.zeros([CD[ln]._Nx, numSimSteps])

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
                TG[ln].step_forward(ang[ln], drv[ln], px[ln], TG[ln]._context[k])
            
            us[ln][:,t], ys[ln][:,t+1] = \
                CD[ln].step_forward(ys[ln][:,t], angleTG[ln,:,k], angleTG[ln,:,kn],
                                    drvTG[ln,:,k]/ctrlTsRatio, drvTG[ln,:,kn]/ctrlTsRatio, dists[ln][:,t])
        
# True angles sampled at Ts
angle    = np.zeros((nLegs, dofTG, numTGSteps))
downSamp = list(range(ctrlTsRatio-1, numSimSteps, ctrlTsRatio))

for ln, leg in enumerate(legs):
    legPos    = int(leg[-1])
    angle[ln,:,:] = angleTG[ln,:,:] + ctrl_to_tg(ys[ln][0:CD[ln]._Nu,downSamp], legPos)
    
matplotlib.use('Agg')
angs           = angle.reshape(-1, angle.shape[-1]).T
angNames       = [(leg + ang) for leg in legs for ang in anglesTG]
pose_3d        = angles_to_pose_names(angs, angNames)
make_fly_video(pose_3d, 'vids/multileg_3layer.mp4')

