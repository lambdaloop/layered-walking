#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys

from tools.ctrl_tools import ControlAndDynamics
from tools.trajgen_tools import TrajectoryGenerator
from tools.angle_functions import anglesTG, anglesCtrl, mapTG2Ctrl, \
                            ctrl_to_tg, tg_to_ctrl
from tools.dist_tools import *

# Usage: python3 main_tg_ctrl_dist.py <leg>
################################################################################
# User-defined parameters
################################################################################
filename = '/home/lisa/Downloads/walk_sls_legs_8.pickle'
#filename = '/home/pierre/data/tuthill/models/models_sls/walk_sls_legs_8.pickle'

numTGSteps  = 200   # How many timesteps to run TG for
Ts          = 1/300 # Sampling time
ctrlTsRatio = 5     # Controller will sample at Ts / ctrlTsRatio

# LQR penalties
drvPen = {'L1': 1e-2, # Looks gait-like but different from pure TG
          'L2': 1e-2, # OK
          'L3': 1e-2, # OK
          'R1': 1e-2, # Looks gait-like but different from pure TG
          'R2': 1e-2, # A bit off pure TG
          'R3': 1e-2  # OK
         }
anglePen = 1e0
inputPen = 1e-8

leg = sys.argv[1]

################################################################################
# Trajectory generator + ctrl and dynamics
################################################################################
legPos  = int(leg[-1])
dofTG   = len(anglesTG)
TG      = TrajectoryGenerator(filename, leg, dofTG, numTGSteps)

numSimSteps = numTGSteps*ctrlTsRatio
CD          = ControlAndDynamics(leg, anglePen, drvPen[leg], inputPen, Ts/ctrlTsRatio)

np.random.seed(623) # For perturbations generated randomly

# Slippery surface
maxVelocity = 15
distsTrue   = get_dists_slippery(maxVelocity, numSimSteps)[leg]

'''
# Uneven surface
maxHt     = 0.05/1000
distsTrue = get_dists_uneven(maxHt, numSimSteps)[leg]
'''

'''
# Stepping on a bump (+ve) or in a pit (-ve) for the second quarter of the simulation
height    = 0.01/1000
distsBump = get_dists_bump_or_pit(height, leg, numSimSteps, 250, 500)[leg]
'''

'''
# Walking on an incline (+ve) or decline (-ve)
angle     = np.radians(30)
distsTrue = get_dists_incline_or_decline(angle, numSimSteps)[leg]
'''

'''
# Leg is missing
missingLeg = 'L1'
distsTrue  = get_dists_missing_leg(missingLeg, numSimSteps)[leg]
'''

distsZero = np.zeros([CD._Nx, numSimSteps])
angleTGDist, drvTGDist, ysDist = CD.run(TG, TG._context, numTGSteps, ctrlTsRatio, distsTrue)
angleTG, drvTG, ys             = CD.run(TG, TG._context, numTGSteps, ctrlTsRatio, distsZero)


# True angle + derivative (sampled at Ts)
dof       = CD._Nu
downSamp  = list(range(ctrlTsRatio-1, numSimSteps, ctrlTsRatio))
angle     = angleTG + ctrl_to_tg(ys[0:dof,downSamp], legPos)
drv       = drvTG + ctrl_to_tg(ys[dof:,downSamp]*CD._Ts, legPos)
angleDist = angleTGDist + ctrl_to_tg(ysDist[0:dof,downSamp], legPos)
drvDist   = drvTGDist + ctrl_to_tg(ysDist[dof:,downSamp]*CD._Ts, legPos)

time = np.array(range(numTGSteps))
plt.figure(1)
plt.clf()
for i in range(dof):
    plt.subplot(2,dof,i+1)
    plt.title(anglesCtrl[legPos][i])
    idx = mapTG2Ctrl[legPos][i]
    
    plt.plot(time, angleTG[idx,:], 'b', label=f'2LayerTG')
    plt.plot(time, angle[idx,:], 'g--', label=f'2Layer')
    plt.plot(time, angleTGDist[idx,:], 'r', label=f'2LayerTG-Dist')
    plt.plot(time, angleDist[idx,:], 'm--', label=f'2Layer-Dist')
    
    plt.subplot(2,dof,i+dof+1)
    plt.title('Velocity')
    plt.plot(time, drvTG[idx,:], 'b', label=f'2LayerTG')
    plt.plot(time, drv[idx,:], 'g--', label=f'2Layer')
    plt.plot(time, drvTGDist[idx,:], 'r', label=f'2LayerTG-Dist')
    plt.plot(time, drvDist[idx,:], 'm--', label=f'2Layer-Dist')

plt.legend()
plt.show()


