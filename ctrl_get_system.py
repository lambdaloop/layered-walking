#!/usr/bin/env python

from ctrl_tools import *
from angle_functions import *
from trajgen_tools import anglesCtrl

#############################################################
# User defined parameters
#############################################################
# Lengths are in milimeters
lCoxa  = all_lengths[0][0]
lFemur = all_lengths[0][1]
lTT    = all_lengths[0][2] + all_lengths[0][3]

# 0.7 mg body mass female, assume leg mass is 1/3 of total body mass
massPerLeg = 0.7 / 1000 / 3 / 6 
mCoxa      = massPerLeg / 4
mFemur     = massPerLeg / 4
mTT        = massPerLeg / 2

# Order: alpha, a, d, theta
DHTable = [('-pi/2',    0,       0, 'q1'),
           ( 'pi/2',    0,   lCoxa, 'q2'),
           ('-pi/2',    0,       0, 'q3'),
           ( 'pi/2',    0,  lFemur, 'q4'),
           (      0,  lTT,       0, 'q5')]

#############################################################
linkLengths = [0, lCoxa, 0, lFemur, lTT]
linkMasses  = [0, mCoxa, 0, mFemur, mTT]

xEqm = []
for angle in anglesCtrl: # Equilibrium joint angles (average)
    xEqm.append(np.radians(median_angles[name_to_index[angle]]))

xEqm += [0, 0, 0, 0, 0]  # Equilibrium joint angular velocity

getLinearizedSystem(DHTable, linkLengths, linkMasses, xEqm)
