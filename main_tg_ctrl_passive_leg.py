#!/usr/bin/env python

import math
import matplotlib.pyplot as plt
import numpy as np
import sys

from tools.ctrl_tools import ControlAndDynamics
from tools.trajgen_tools import TrajectoryGenerator, WalkingData
from tools.angle_functions import anglesTG, anglesCtrl, \
                                  ctrl_to_tg, tg_to_ctrl, legs, \
                                  default_angles
from tools.dist_tools import *
from tools.ground_model import GroundModel

# Usage: python3 main_tg_ctrl_dist.py <leg>
################################################################################
# User-defined parameters
################################################################################
# filename = '/home/lisa/Downloads/walk_sls_legs_subang_1.pickle'
filename = '/home/lili/data/tuthill/models/models_sls/walk_sls_legs_subang_1.pickle'

walkingSettings = [14, 0, 0] # walking, turning, flipping speeds (mm/s)

numTGSteps      = 200   # How many timesteps to run TG for
Ts              = 1/300 # How fast TG runs
ctrlSpeedRatio  = 2     # Controller will run at Ts / ctrlSpeedRatio
ctrlCommRatio   = 8     # Controller communicates to TG this often (as multiple of Ts)
actDelay        = 0.03  # Seconds; typically 0.02-0.04
senseDelay      = 0.00  # Seconds; typically 0.01

leg = 'R1'

################################################################################
# Get walking data
################################################################################
wd       = WalkingData(filename)
bout     = wd.get_bout(walkingSettings, offset=4)

# Use constant contexts
context  = np.array(walkingSettings).reshape(1,3)
contexts = np.repeat(context, numTGSteps, axis=0)

offset = 48
angInit   = bout['angles'][leg][offset]
drvInit   = bout['derivatives'][leg][offset]
phaseInit = bout['phases'][leg][offset]

tarsus_ratio = 0.05

################################################################################
# Ground model
################################################################################
# Assuming flat ground
gndHeight = -0.8

################################################################################
# Trajectory generator alone
################################################################################
ground_tg    = GroundModel(offset=[0, 0, gndHeight], phi=0, theta=0)

TGNoGround = TrajectoryGenerator(filename, leg, numTGSteps, groundModel=None)
TG         = TrajectoryGenerator(filename, leg, numTGSteps, groundModel=ground_tg)

angleTGng, drvTGng, phaseTGng = TGNoGround.get_future_traj(0, numTGSteps,
                                    np.copy(angInit), np.copy(drvInit), phaseInit, contexts)

angleTG, drvTG, phaseTG = TG.get_future_traj(0, numTGSteps,
                              np.copy(angInit), np.copy(drvInit),
                                             phaseInit, contexts)


################################################################################
# TG plus controller and dynamics
################################################################################
dAct   = int(actDelay / Ts * ctrlSpeedRatio)
dSense = int(senseDelay / Ts * ctrlSpeedRatio)
print(f'Steps of actuation delay: {dAct}')
print(f'Steps of sensory delay  : {dSense}')

ground    = GroundModel(offset=[0, 0, gndHeight], phi=0, theta=0, tarsus=True)

TG = TrajectoryGenerator(filename, leg, numTGSteps, groundModel=ground,
                                 tarsus_ratio=tarsus_ratio)


legPos  = int(leg[-1])
numAng  = TG._numAng

namesTG = [x[2:] for x in TG._angle_names]
CD      = ControlAndDynamics(leg, Ts/ctrlSpeedRatio, dSense, dAct, namesTG)

legIdx         = legs.index(leg)
fullAngleNames = [(leg + ang) for ang in namesTG]
dof            = CD._Nur

numSimSteps   = numTGSteps*ctrlSpeedRatio
angleTG2      = np.zeros((numAng+1, numTGSteps))
drvTG2        = np.zeros((numAng, numTGSteps))
phaseTG2      = np.zeros(numTGSteps)
angleTG2[:numAng,0], drvTG2[:numAng,0], phaseTG2[0] = angInit, drvInit, phaseInit

xs    = np.zeros([CD._Nx, numSimSteps])
xEsts = np.zeros([CD._Nx, numSimSteps])
us    = np.zeros([CD._Nu, numSimSteps])

lookahead = math.ceil(dAct/ctrlSpeedRatio)
positions = np.zeros([5, 3, numSimSteps])

angTarsus = np.zeros(numSimSteps)
angTarsus[0] = default_angles[leg + 'D_flex']
angleTG2[-1, 0] = angTarsus[0]

# No disturbances for now
dist = np.zeros(CD._Nxr)

# numSimSteps = 91

for t in range(numSimSteps-1):
# for t in range(140 - 1):
    k  = int(t / ctrlSpeedRatio)     # Index for TG data
    kn = int((t+1) / ctrlSpeedRatio) # Next index for TG data

    ang   = angleTG2[:numAng,k] + ctrl_to_tg(xs[0:dof,t], legPos, namesTG)

    if not (k % ctrlCommRatio) and k != kn and k < numTGSteps-1:
        ang = angleTG2[:numAng,k] + ctrl_to_tg(xs[0:dof,t], legPos, namesTG)
        drv = drvTG2[:numAng,k] + ctrl_to_tg(xs[dof:dof*2,t]*CD._Ts, legPos, namesTG)

        kEnd = min(k+ctrlCommRatio+lookahead, numTGSteps-1)
        angleTG2[:,k+1:kEnd+1], drvTG2[:numAng,k+1:kEnd+1], phaseTG2[k+1:kEnd+1] = \
            TG.get_future_traj(k, kEnd, ang, drv, phaseTG2[k], contexts, tarsus=angTarsus[t])

    k1 = min(int((t+dAct) / ctrlSpeedRatio), numTGSteps-1)
    k2 = min(int((t+dAct+1) / ctrlSpeedRatio), numTGSteps-1)

    anglesAhead = np.concatenate((angleTG2[:numAng,k1].reshape(numAng,1),
                                  angleTG2[:numAng,k2].reshape(numAng,1)), axis=1)
    drvsAhead   = np.concatenate((drvTG2[:,k1].reshape(numAng,1),
                                  drvTG2[:,k2].reshape(numAng,1)), axis=1)/ctrlSpeedRatio
    dist = np.zeros(CD._Nxr)
    us[:,t], xs[:,t+1], xEsts[:,t+1] = CD.step_forward(xs[:,t], xEsts[:,t], anglesAhead, drvsAhead, dist)

    # Ground model
    ang_prev = angleTG2[:numAng,k] + ctrl_to_tg(xs[0:dof,t], legPos, namesTG)
    ang      = angleTG2[:numAng,kn] + ctrl_to_tg(xs[0:dof,t+1], legPos, namesTG)
    drv      = drvTG2[:numAng,kn] + ctrl_to_tg(xs[dof:dof*2,t+1]*CD._Ts, legPos, namesTG)

    # implement hook's law here
    med = default_angles[leg + 'D_flex']
    angTarsus[t+1] = angTarsus[t] + (med - angTarsus[t]) * tarsus_ratio

    ang_prev = np.append(ang_prev, angTarsus[t])
    ang = np.append(ang, angTarsus[t+1])

    ang_new_dict, drv_new_dict, ground_legs = ground.step_forward(
        {leg: ang_prev}, {leg: ang}, {leg: drv})

    ang_next = ang_new_dict[leg]
    drv_next = drv_new_dict[leg]

    angTarsus[t+1] = ang_next[-1]

    if leg in ground_legs:
        dist = np.zeros(CD._Nxr)
        dist[0:dof] = tg_to_ctrl(ang_next - ang, legPos, namesTG)
        # dist[dof:dof*2] = tg_to_ctrl(0 - drv, legPos, namesTG)
        # dist[dof:dof*2] = tg_to_ctrl(ang_next - angleTG2[:numAng,kn], legPos, namesTG) / (Ts * 2)
        us[:,t], xs[:,t+1], xEsts[:,t+1] = CD.step_forward(xs[:,t], xEsts[:,t], anglesAhead, drvsAhead, dist)
        # angleChange = ang_next - angleTG[:numAng,kn]
        print(f'time: {k}')
        # print(f'time: {k}, angle change (deg): {angleChange}')
        # xs[0:dof,t+1] = tg_to_ctrl(ang_next - angleTG2[:numAng,kn], legPos, namesTG)
        # xs[0:dof,t+1] = tg_to_ctrl(drv_next - drvTG2[:numAng,kn], legPos, namesTG)

# plt.figure(2)
# plt.clf()
# # plt.imshow(xEsts - xs, aspect='auto')
# t = np.arange(xs.shape[1]) / 2
# # plt.plot(t, xs[dof])
# plt.subplot(3, 1, 1)
# plt.plot(CD._A[dof])
# plt.title('weights for derivative of A_abduct')
# plt.subplot(3, 1, 2)
# plt.plot(xs[:, 13])
# plt.title('full state')
# plt.subplot(3, 1, 3)
# plt.plot(CD._A[dof] * xs[:, 13])
# plt.title('weights for derivative of A_abduct * full state')
# # plt.imshow(CD._A)
# plt.draw()
# plt.show(block=False)

plt.figure(2)
plt.clf()
plt.plot(angTarsus)
plt.draw()
plt.show(block=False)





################################################################################
# Postprocessing and plotting
################################################################################
downSamp = list(range(0, numSimSteps, ctrlSpeedRatio))
angle2   = angleTG2[:numAng] + ctrl_to_tg(xs[0:dof,downSamp], legPos, namesTG)
drv2     = drvTG2 + ctrl_to_tg(xs[dof:dof*2,downSamp]*CD._Ts, legPos, namesTG)
angTarsus2 = angTarsus[downSamp]

heightsTGng = np.zeros(numTGSteps)
heightsTG   = np.zeros(numTGSteps)
heightsTG2  = np.zeros(numTGSteps)
heights2    = np.zeros(numTGSteps)

gContactTGng = np.array([None] * numTGSteps)
gContactTG   = np.array([None] * numTGSteps)
gContactTG2  = np.array([None] * numTGSteps)
gContact2    = np.array([None] * numTGSteps)

for t in range(numTGSteps):
    # need to modify to append tarsus here
    heightsTGng[t] = ground_tg.get_positions[leg](angleTGng[:,t])[-1, 2]
    heightsTG[t]   = ground_tg.get_positions[leg](angleTG[:,t])[-1, 2]
    heightsTG2[t]  = ground.get_positions[leg](angleTG2[:,t])[-1, 2]
    heights2[t]    = ground.get_positions[leg](np.append(angle2[:,t], angTarsus2[t]))[-1, 2]

    if heightsTGng[t] <= gndHeight:
        gContactTGng[t] = heightsTGng[t]
    if heightsTG[t] <= gndHeight:
        gContactTG[t] = heightsTG[t]
    if heightsTG2[t] <= gndHeight:
        gContactTG2[t] = heightsTG2[t]
    if heights2[t] <= gndHeight:
        gContact2[t] = heights2[t]

mapIx = [namesTG.index(n) for n in anglesCtrl[legPos]]
time  = np.array(range(numTGSteps))

plt.figure(1)
plt.clf()

for i in range(dof):
    plt.subplot(3,dof,i+1)
    plt.title(anglesCtrl[legPos][i])
    idx = mapIx[i]

    plt.plot(time, np.mod(angleTGng[idx,:], 360), 'b', label=f'TG No Ground')
    plt.plot(time, np.mod(angleTG[idx,:], 360), 'g', label=f'TG Ground')
    plt.plot(time, np.mod(angleTG2[idx,:], 360), 'r', label=f'2Layer TG')
    plt.plot(time, np.mod(angle2[idx,:], 360), 'k--', label=f'2Layer')
    # plt.ylim(-180, 180)
    plt.ylim(0, 360)

    if i==0:
        plt.legend()

    plt.subplot(3,dof,i+dof+1)
    plt.title('Velocity')
    plt.plot(time, drvTGng[idx,:], 'b')
    plt.plot(time, drvTG[idx,:], 'g')
    plt.plot(time, drvTG2[idx,:], 'r')
    plt.plot(time, drv2[idx,:], 'k--')

    plt.subplot(3,1,3)
    plt.title('Height')
    plt.plot(time, heightsTGng, 'b')
    plt.plot(time, heightsTG, 'g')
    plt.plot(time, heightsTG2, 'r')
    plt.plot(time, heights2, 'k--')

    plt.plot(time, gContactTGng, 'b*', markersize=6)
    plt.plot(time, gContactTG, 'g*', markersize=6)
    plt.plot(time, gContactTG2, 'r*', markersize=6)
    plt.plot(time, gContact2, 'k*', markersize=6)

plt.draw()
plt.show(block=False)
