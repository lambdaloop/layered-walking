#!/usr/bin/env python

from ctrl_tools import *
from angle_functions import *

################################################################################
# Parameters / DH conventions
################################################################################
# 0.7 mg body mass female, assume leg mass is 1/3 of total body mass
# and each segment is 1/4 of total leg mass
massPerLeg = 0.7 / 1000 / 3 / 6 
mCoxa      = massPerLeg / 4
mFemur     = massPerLeg / 4
mTT        = massPerLeg / 2

legs_to_compute = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3'] # Which legs to get a system for

def get_robot_params(leg):
    ''' 
    Given leg (e.g. 'L1'), return (DHTable, linkLengths, linkMasses, xEqm)
    DHTable     : list of tuples (alpha, a, d, theta) representing DH params
    linkLengths : list of lengths of links
    linkMasses  : list of masses of links
    xEqm        : equilibrium point (angles and angular velocity)
    '''
    legIdx = legs.index(leg)
    legPos = int(leg[-1])

    # Set equilibrium as average joint angle with zero velocity
    xEqm = []
    for angle in anglesCtrl[legPos]:
        angName = leg + angle
        xEqm.append(np.radians(median_angles[name_to_index[angName]]))
    xEqm += [0] * len(xEqm) # Angular velocity entires are zero

    # Get lengths (convert from millimeters)
    lCoxa  = all_lengths[legIdx][0] / 1000
    lFemur = all_lengths[legIdx][1] / 1000
    lTT    = (all_lengths[legIdx][2] + all_lengths[legIdx][3]) / 1000
    
    # Calculate inertias
    iCoxa  = 1.0/3.0 * mCoxa * lCoxa * lCoxa
    iFemur = 1.0/3.0 * mFemur * lFemur * lFemur
    iTT    = 1.0/3.0 * mTT * lTT * lTT

    # Get DH table, link masses
    if legPos == 1: # Front legs
        DHTable = [('pi/2',  0,       0,      'q1'),
                   ('-pi/2', 0,       lCoxa,  'q2'),
                   (0,       lFemur,  0,      'q3'),
                   (0,       lTT,     0,      'q4')]
        linkMasses  = [0, mCoxa, mFemur, mTT]
    elif legPos == 2: # Middle legs
        DHTable = [('pi/2',  0,    0,      'q1'),
                   ('-pi/2', 0,    lFemur, 'q2'),
                   (0,       lTT,  0,      'q3')]
        linkMasses  = [0, mFemur, mTT]
    else: # Hind legs
        DHTable = [(0, lFemur,  0, 'q1'),
                   (0, lTT,     0, 'q2')]
        linkMasses  = [mFemur, mTT]
    
    # Get inertias
    dof      = len(DHTable)
    inertias = [None] * dof
    for i in range(dof):
        inertias[i] = [0] * 6 # Convention: [L_xx, L_xy, L_xz, L_yy, L_yz, L_zz]

    if legPos == 1:
        inertias[1][3] = iCoxa  # L2_yy
        inertias[2][5] = iFemur # L3_zz
        inertias[3][5] = iTT    # L4_zz
    elif legPos == 2: 
        inertias[1][3] = iFemur # L2_yy
        inertias[2][5] = iTT    # L3_zz
    else:
        inertias[0][5] = iFemur # L1_zz
        inertias[1][5] = iTT    # L2_zz

    return (DHTable, linkMasses, inertias, xEqm)

################################################################################
# Compute linearized system
################################################################################
for leg in legs_to_compute:
    print('Calculating for leg ' + leg)
    (DHTable, linkMasses, inertias, xEqm) = get_robot_params(leg)
    get_linearized_system(DHTable, linkMasses, inertias, xEqm, leg)


