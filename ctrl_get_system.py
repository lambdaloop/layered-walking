#!/usr/bin/env python

from ctrl_tools import *
from angle_functions import *

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

# Note: sympy trig functions take radians
angles_ctrl = ['L1A_abduct', 'L1B_flex', 'L1A_rot', 'L1C_flex']

# body-coxa flexion, A_abduct
bcFlexAvg = np.radians(median_angles[angles_ctrl[0]]) 

# coxa-femur flexion, B_flex
cfFlexAvg = np.radians(median_angles[angles_ctrl[1]])  

# coxa rotation, A_rot
cfRotAvg  = np.radians(median_angles[angles_ctrl[2]]) 

# femur-tibia flexion, C_flex
ftFlexAvg = np.radians(median_angles[angles_ctrl[3]]) 

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

getLinearizedSystem(DHTable, linkLengths, linkMasses, xEqm)
