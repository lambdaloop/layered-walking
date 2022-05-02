#!/usr/bin/env python

import numpy as np
from tools import *

#############################################################
# User defined parameters
#############################################################
# TODO: need to get realistic valuse of these
lCoxa  = 1
lFemur = 1
lTT    = 1 

mCoxa  = 1
mFemur = 1
mTT    = 1

# Sampling time
Ts = 1/300 # 300 times per second

# Note: sympy trig functions take radians
bcFlexAvg = np.radians(150) # body-coxa flexion
cfFlexAvg = np.radians(60)  # coxa-femur flexion
cfRotAvg  = np.radians(165) # coxa-femur rotation
ftFlexAvg = np.radians(95)  # femur-tibia flexion

# Order: alpha, a, d, theta
DHTable = [(      0, lCoxa,      0, 'q1'),
           ('-pi/2',     0,      0, 'q2'),
           ( 'pi/2',     0, lFemur, 'q3'),
           (      0,  lTT,       0, 'q4')]

#############################################################

linkLengths = [lCoxa, 0, lFemur, lTT]
linkMasses  = [mCoxa, 0, mFemur, mTT]

dof  = len(linkLengths)
xEqm = [bcFlexAvg, cfFlexAvg, cfRotAvg, ftFlexAvg, 0, 0, 0, 0]

# Only need to do this once
# getLinearizedSystem(DHTable, linkLengths, linkMasses, xEqm)

from tools import *

# TODO: can separate this into a different file (generation vs analysis)
ALin, BLin, uEqm = loadLinearizedSystem()

Nx = 2*dof
Nu = dof

# Discretize
A = np.eye(Nx) + ALin*Ts
B = BLin*Ts

# Controller
import control

# TODO: here

# Check controllability
Qc = control.ctrb(A,B)
rankCtrb = np.linalg.matrix_rank(Qc)
if rankCtrb != Nx:
    print('System uncontrollable!')

# Check open loop eigenvalues
eigsOL    = np.linalg.eig(A)[0]
specRadOL = max(np.abs(eigsOL))

# Get controller
anglePenalty    = 1e8
velocityPenalty = 1e-8
inputPenalty    = 1e-8

Q1 = anglePenalty*np.eye(dof)
Q2 = velocityPenalty*np.eye(dof)

from scipy.linalg import block_diag
Q = block_diag(Q1, Q2)
R = inputPenalty*np.eye(Nu)

K         = control.dlqr(A, B, Q, R)[0]
ACL       = A - B @ K
eigsCL    = np.linalg.eig(ACL)[0]
specRadCL = max(np.abs(eigsCL))
if specRadCL >= 1:
    print('Controller did not stabilize!')

tHorizon = 100
time = np.array(range(tHorizon))

xEqm = [bcFlexAvg, cfFlexAvg, cfRotAvg, ftFlexAvg, 0, 0, 0, 0]
xEqm = np.array(xEqm)
ys         = np.zeros([Nx, tHorizon])
us         = np.zeros([Nu, tHorizon])
trajs      = np.zeros([Nx, tHorizon])

for joint in range(dof):
    trajs[joint,:] = 0.5 * np.sin(time * 0.25) + xEqm[joint]

for t in range(tHorizon-1):
    wtraj        = A @ (trajs[:,t] - xEqm) + xEqm - trajs[:,t+1]    
    ys[:,t+1]    = ACL @ ys[:,t] + wtraj
    us[:,t]      = K @ ys[:,t]

qs = ys + trajs

import matplotlib.pyplot as plt

for pltState in range(4):
    plt.subplot(2,2,pltState+1)
    plt.plot(time, np.degrees(qs[pltState,:]), '--', label=f'q{pltState}')
    plt.plot(time, np.degrees(trajs[pltState,:]), label=f'traj{pltState}')

plt.legend()
plt.show()


