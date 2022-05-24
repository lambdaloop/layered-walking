#!/usr/bin/env python

from ctrl_tools import *
from angle_functions import *
from trajgen_tools import anglesCtrl1

#############################################################
# User defined parameters
#############################################################
# Lengths are in milimeters
lCoxa  = all_lengths[0][0] / 1000
lFemur = all_lengths[0][1] / 1000
lTT    = (all_lengths[0][2] + all_lengths[0][3]) / 1000

# 0.7 mg body mass female, assume leg mass is 1/3 of total body mass
massPerLeg = 0.7 / 1000 / 3 / 6 
mCoxa      = massPerLeg / 4
mFemur     = massPerLeg / 4
mTT        = massPerLeg / 2

# Inertias
iCoxa  = 1.0/3.0 * mCoxa * lCoxa * lCoxa
iFemur = 1.0/3.0 * mFemur * lFemur * lFemur
iTT    = 1.0/3.0 * mTT * lTT * lTT

# Order: alpha, a, d, theta
DHTable = [('pi/2',  0,       0,      'q1'),
           ('-pi/2', 0,       lCoxa,  'q2'),
           (0,       lFemur,  0,      'q3'),
           (0,       lTT,     0,      'q4')]

dof = len(DHTable)

inertias = [None] * dof
for i in range(dof):
    inertias[i] = [0] * 6 # Convention: [L_xx, L_xy, L_xz, L_yy, L_yz, L_zz]
inertias[1][3] = iCoxa # L2_yy
inertias[2][5] = iFemur # L3_zz
inertias[3][5] = iTT # L4_zz

#############################################################
linkMasses  = [0, mCoxa, mFemur, mTT]

xEqm = []
for angle in anglesCtrl1: # Equilibrium joint angles (average)
    xEqm.append(np.radians(median_angles[name_to_index[angle]]))

xEqm += [0] * len(xEqm)

getLinearizedSystem(DHTable, linkMasses, inertias, xEqm)
