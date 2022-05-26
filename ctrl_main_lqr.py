#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ctrl_tools import *
from trajgen_tools import *
from angle_functions import angles_to_pose_names, make_fly_video, legs, anglesTG

################################################################################
# User-defined parameters
################################################################################
numSimSteps = 200 # How many timesteps to run model for
Ts          = 1/300 # Sampling time



################################################################################
# Solo TG
################################################################################
filename = '/home/lisa/Downloads/walk_sls_legs_2.pickle'
# filename = '/home/pierre/data/tuthill/models/models_sls/walk_sls_legs_5.pickle'

def kuramato_deriv(px, alphas, offsets, ws):
    return ws + np.array([
        np.sum(alphas[i] * np.sin(px - px[i] - offsets[i]))
        for i in range(len(px))
    ])

offsets = np.array([
    [0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0],
])*np.pi

alphas = np.ones((6,6))*4.0

n_legs = len(legs)
dofTG   = len(anglesTG)
TG = [None for i in range(n_legs)]
angleTG = np.zeros((n_legs, dofTG, numSimSteps))
drvTG   = np.zeros((n_legs, dofTG, numSimSteps))
phaseTG = np.zeros((n_legs, numSimSteps))

for legnum, leg in enumerate(legs):
    TG[legnum] = TrajectoryGenerator(filename, leg, dofTG, numSimSteps)
    angleTG[legnum,:,0], drvTG[legnum,:,0], phaseTG[legnum,0] = TG[legnum].get_initial_vals()


'''
for t in range(numSimSteps-1):
    ws = np.zeros(6)
    px = phaseTG[:, t]
    px_half = px + 0.5*Ts * kuramato_deriv(px, alphas, offsets, ws)
    px = px + Ts * kuramato_deriv(px_half, alphas, offsets, ws)

    for legnum in range(n_legs):
        angleTG[legnum, :,t+1], drvTG[legnum, :,t+1], phaseTG[legnum, t+1] = \
            TG[legnum].step_forward(angleTG[legnum, :,t], drvTG[legnum, :,t],
                                    px[legnum], TG[legnum]._context[t])

matplotlib.use('Agg')
angs           = angleTG.reshape(-1, angleTG.shape[-1]).T
angNames       = [(leg + ang) for leg in legs for ang in anglesTG]
pose_3d        = angles_to_pose_names(angs, angNames)
make_fly_video(pose_3d, 'vids/multileg_twolayer_tgonly.mp4')
'''
################################################################################
# 2-Layer: TG + Controller + Dynamics
################################################################################
# LQR penalties
anglePen = 1e0
drvPen1  = 5e1 # for R1 and L1
drvPen2  = 1e0 # for others
inputPen = 1e-8

CD = [None for i in range(n_legs)]

angleTG2 = np.zeros((n_legs, dofTG, numSimSteps))
drvTG2   = np.zeros((n_legs, dofTG, numSimSteps))
phaseTG2 = np.zeros((n_legs, numSimSteps))
ang = np.zeros((n_legs, dofTG))
drv = np.zeros((n_legs, dofTG))
phase = np.zeros(n_legs)

ys = [None for i in range(n_legs)]
us = [None for i in range(n_legs)]

for legnum, leg in enumerate(legs):
    if leg == 'L1' or leg == 'R1':
        CD[legnum] = ControlAndDynamics(leg, anglePen, drvPen1, inputPen, Ts)
    else:
        CD[legnum] = ControlAndDynamics(leg, anglePen, drvPen2, inputPen, Ts)
    
    dof      = CD[legnum]._Nu
    ys[legnum]       = np.zeros([CD[legnum]._Nx, numSimSteps])
    us[legnum]       = np.zeros([CD[legnum]._Nu, numSimSteps])

    ang[legnum], drv[legnum], phase[legnum] = TG[legnum].get_initial_vals()
    angleTG2[legnum,:,0], drvTG2[legnum,:,0], phaseTG2[legnum,0] = ang[legnum], drv[legnum], phase[legnum]


# Simulation
for t in range(numSimSteps-1):
    ws = np.zeros(6)
    px = phaseTG2[:, t]
    px_half = px + 0.5*Ts * kuramato_deriv(px, alphas, offsets, ws)
    px = px + Ts * kuramato_deriv(px_half, alphas, offsets, ws)

    for legnum, leg in enumerate(legs):
        legPos   = int(leg[-1])
        dof      = CD[legnum]._Nu
        
        angleTG2[legnum, :,t+1], drvTG2[legnum, :,t+1], phaseTG2[legnum, t+1] = \
            TG[legnum].step_forward(ang[legnum], drv[legnum], px[legnum], TG[legnum]._context[t])
        
        us[legnum][:,t], ys[legnum][:,t+1] = CD[legnum].step_forward(ys[legnum][:,t], angleTG2[legnum,:,t], angleTG2[legnum,:,t+1],
                                     drvTG2[legnum,:,t], drvTG2[legnum,:,t+1])
        
        ang[legnum] = angleTG2[legnum,:,t+1] + ctrl_to_tg(ys[legnum][0:dof,t+1], legPos)    
        drv[legnum] = drvTG2[legnum,:,t+1]   + ctrl_to_tg(ys[legnum][dof:,t+1]*Ts, legPos)


angle2 = np.zeros((n_legs, dofTG, numSimSteps))
for legnum, leg in enumerate(legs):
    legPos         = int(leg[-1])
    dof      = CD[legnum]._Nu
    angle2[legnum] = angleTG2[legnum] + ctrl_to_tg(ys[legnum][0:dof,:], legPos)

## TODO
matplotlib.use('Agg')
angs           = angle2.reshape(-1, angle2.shape[-1]).T
angNames       = [(leg + ang) for leg in legs for ang in anglesTG]
pose_3d        = angles_to_pose_names(angs, angNames)
make_fly_video(pose_3d, 'vids/multileg_3layer.mp4')



'''
for t in range(numSimSteps-1):
    #angleTG2[:,t+1], drvTG2[:,t+1], phaseTG2[t+1] = \
    #    TG.step_forward(ang, drv, phaseTG2[t], TG._context[t])
    
    #us[:,t], ys[:,t+1] = CD.step_forward(ys[:,t], angleTG2[:,t], angleTG2[:,t+1],
    #                                     drvTG2[:,t], drvTG2[:,t+1])
    
    ang = angleTG2[:,t+1] + ctrl_to_tg(ys[0:dof,t+1], legPos)    
    drv = drvTG2[:,t+1]   + ctrl_to_tg(ys[dof:,t+1]*Ts, legPos)
'''





'''    
angleErr = np.linalg.norm(np.degrees(ys[0:dof,]), ord='fro')
drvErr   = np.linalg.norm(np.degrees(ys[dof:,]*Ts), ord='fro')

print(f'Frob norm of angle error: {angleErr} deg')
print(f'Frob norm of angular velocity error: {drvErr} deg/s')

angle2 = tg_to_ctrl(angleTG2, legPos) + ys[0:dof,:]
drv2   = tg_to_ctrl(drvTG2, legPos)   + ys[dof:,:]*Ts
time   = np.array(range(numSimSteps))

plt.figure(1)
plt.clf()
for i in range(dof):
    plt.subplot(2,dof,i+1)
    plt.title(anglesCtrl[legPos][i])
    idx = mapTG2Ctrl[legPos][i]
    
    plt.plot(time, angleTG[idx,:], 'b', label=f'SoloTG')
    plt.plot(time, angleTG2[idx,:], 'r', label=f'2LayerTG')
    plt.plot(time, np.degrees(angle2[i,:]), 'k--', label=f'2Layer')
    
    plt.subplot(2,dof,i+dof+1)
    plt.title('Velocity')
    plt.plot(time, drvTG[idx,:], 'b', label=f'SoloTG')
    plt.plot(time, drvTG2[idx,:], 'r', label=f'2LayerTG')
    plt.plot(time, np.degrees(drv2[i,:]), 'k--', label=f'2Layer')
plt.legend()
plt.show()

if makeVideo:
    matplotlib.use('Agg')
    angs           = np.degrees(angle2).T
    angNames       = [(leg + ang) for ang in anglesCtrl[legPos]]
    pose_3d        = angles_to_pose_names(angs, angNames)
    pose_3d[:, 1:] = np.nan
    make_fly_video(pose_3d, 'vids/' + leg + '_twolayer.mp4')
'''


