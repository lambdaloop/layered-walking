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

basename = 'test_ground'

################################################################################
# User-defined parameters
################################################################################
filename = '/home/lisa/Downloads/walk_sls_legs_subang_1.pickle'

walkingSettings = [10, 0, 0] # walking, turning, flipping speeds (mm/s)

numTGSteps      = 200   # How many timesteps to run TG for
Ts              = 1/300 # How fast TG runs
ctrlSpeedRatio  = 2     # Controller will run at Ts / ctrlSpeedRatio
ctrlCommRatio   = 8     # Controller communicates to TG this often (as multiple of Ts)
actDelay        = 0.03  # Seconds; typically 0.02-0.04
couplingDelay   = 0.010

################################################################################
# Ground model
################################################################################
# TODO: Should we allow this to differ from leg to leg?
boutNum   = 0
startIdx  = 0
gndHeight = -0.85
ground    = GroundModel(offset=[0, 0, gndHeight], phi=0, theta=0)

################################################################################
# Get walking data
################################################################################
wd       = WalkingData(filename)
bout     = wd.get_bout(walkingSettings, offset=boutNum)

# Use constant contexts
context  = np.array(walkingSettings).reshape(1,3)
contexts = np.repeat(context, numTGSteps, axis=0)

angInit   = bout['angles']
drvInit   = bout['derivatives']
phaseInit = bout['phases']

################################################################################
# Setup
################################################################################
dAct   = int(actDelay / Ts * ctrlSpeedRatio)
print(f'Steps of actuation delay: {dAct}')

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
us      = [None for i in range(nLegs)]

fullAngleNames = []

for ln, leg in enumerate(legs):    
    TG[ln] = TrajectoryGenerator(filename, leg, numTGSteps, groundModel=ground)

    fullAngleNames.append(TG[ln]._angle_names)

    namesTG[ln] = [x[2:] for x in TG[ln]._angle_names]
    CD[ln] = ControlAndDynamics(leg, Ts/ctrlSpeedRatio, dAct, namesTG[ln])
    fullAngleNames.append([(leg + ang) for ang in namesTG[ln]])
    dof = TG[ln]._numAng

    angleTG[ln,:dof,0], drvTG[ln,:dof,0], phaseTG[ln,0] = \
        angInit[leg][startIdx], drvInit[leg][startIdx], phaseInit[leg][startIdx]
    
    xs[ln]    = np.zeros([CD[ln]._Nx, numSimSteps])
    us[ln]    = np.zeros([CD[ln]._Nu, numSimSteps])

################################################################################
# Simulation
################################################################################
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
        dof     = TG[ln]._numAng
        
        if not (k % ctrlCommRatio) and k != kn and k < numTGSteps-1:
            ang = angleTG[ln,:dof,k] + ctrl_to_tg(xs[ln][0:CD[ln]._Nur,t], legPos, namesTG[ln])
            drv = drvTG[ln,:dof,k] + ctrl_to_tg(xs[ln][CD[ln]._Nur:CD[ln]._Nur*2,t]*CD[ln]._Ts,
                  legPos, namesTG[ln])

            kEnd = min(k+ctrlCommRatio+lookahead, numTGSteps-1)
            angleTG[ln,:dof,k+1:kEnd+1], drvTG[ln,:dof,k+1:kEnd+1], phaseTG[ln,k+1:kEnd+1] = \
                TG[ln].get_future_traj(k, kEnd, ang, drv, phaseTG[ln,k], contexts)
                
        anglesAhead = np.concatenate((angleTG[ln,:,k1].reshape(dofTG,1),
                                      angleTG[ln,:,k2].reshape(dofTG,1)), axis=1)
        drvsAhead   = np.concatenate((drvTG[ln,:,k1].reshape(dofTG,1),
                                      drvTG[ln,:,k2].reshape(dofTG,1)), axis=1)/ctrlSpeedRatio
        
        n1  = CD[ln]._Nxr*(dAct+1)
        xf  = xs[ln][n1-CD[ln]._Nxr:n1, t]
        ang = angleTG[ln,:dof, k2] + ctrl_to_tg(xf[0:dof], legPos, namesTG[ln])
        
        # Use ground model on future predicted trajectory, to check if it hits ground
        # Slightly hacky: don't use ground model velocity output
        angNew, junk, groundLegs = ground.step_forward({leg: ang}, {leg: ang}, {leg: ang})

        gndAdjust = 0
        if leg in groundLegs: # Future is predicted to hit ground; account for this
            gndAdjust = tg_to_ctrl(angNew[leg] - ang, legPos, namesTG[ln])
            gndAdjust = np.concatenate((gndAdjust, np.zeros(dof)))    
        
        # Zero disturbance for now
        dist = np.zeros(CD[ln]._Nxr)
        us[ln][:,t], xs[ln][:,t+1] = CD[ln].step_forward(
            xs[ln][:,t], anglesAhead, drvsAhead, dist, gndAdjust)

        # Apply ground interaction to dynamics
        # Slightly hacky: don't use ground model velocity output
        ang = angleTG[ln,:dof,kn] + ctrl_to_tg(xs[ln][0:dof,t+1], legPos, namesTG[ln])
        angNew, junk, groundLegs = ground.step_forward({leg: ang}, {leg: ang}, {leg: ang})
        
        if leg in groundLegs: # Treat the ground interaction as a disturbance
            angNxt                = tg_to_ctrl(angNew[leg] - angleTG[ln,:dof,kn], legPos, namesTG[ln])
            groundDist            = np.zeros(dof*2)
            groundDist[0:dof]     = angNxt - xs[ln][0:dof,t+1]
            augDist               = CD[ln].get_augmented_dist(groundDist)
            xs[ln][:,t+1]         += augDist

################################################################################
# Postprocessing and plotting
################################################################################
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
