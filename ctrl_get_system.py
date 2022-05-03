#!/usr/bin/env python

from ctrl_tools import *

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

getLinearizedSystem(DHTable, linkLengths, linkMasses, xEqm)
