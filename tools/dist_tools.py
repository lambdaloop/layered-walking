#!/usr/bin/env python

import numpy as np
from enum import Enum
from scipy.optimize import fsolve
from tools.angle_functions import all_lengths, angles_to_pose_names, \
                                  anglesCtrl, anglesTG, \
                                  get_avg_angles, get_leg_lengths, legs


################################################################################
# Helper functions
################################################################################
def root_target_angles(x, angles, lengths, height):
    ''' This function is used (via fsolve) to find the coxa-femur flexion that 
        corresponds to the desired height (positive = toward body) '''
    l1 = lengths[0]; t1 = angles[0]
    l2 = lengths[1]; t2 = angles[1]
    l3 = lengths[2]; t3 = angles[2]

    a = t1 - np.pi/2
    b = t3 - t2 + a
    c = t3 - x + a    
    return l3*np.cos(b) - l3*np.cos(c) - l2*np.cos(t2-a) + l2*np.cos(x-a) - height        



def get_dists_endeffector_moves(height, leg):
    ''' Positive height = leg moves toward the body '''
    legPos        = int(leg[-1])
    legIdx        = legs.index(leg)
    numAngles     = len(anglesCtrl[legPos])
    avgAngleNames = ['A_abduct', 'B_flex', 'C_flex']
    avgAngles     = get_avg_angles(avgAngleNames, leg)
    
    lCoxa, lFemur, lTT = get_leg_lengths(legIdx)    
    lengths            = [lCoxa, lFemur, lTT]
    
    bFlex     = fsolve(root_target_angles, x0=avgAngles[1], 
                       args=(avgAngles, lengths, height))
    bFlexDiff = bFlex[0] - avgAngles[1]
        
    dists = np.zeros([numAngles*2])
    dists[anglesCtrl[legPos].index('B_flex')] = bFlexDiff
    
    return dists



################################################################################
# Ground contact functions for users
################################################################################
def get_ground_contact_threshold(angles, fullAngleNames, legIdx):
    ''' Expect angles in degrees, in order specified by anglesTG, e.g. angleTG '''
    angs     = angles.reshape(-1, angles.shape[-1]).T
    poses    = angles_to_pose_names(angs, fullAngleNames)

    numSteps       = angles.shape[1] 
    contactHeights = np.array([])

    for t in range(1, numSteps-1):
        lastHeight = poses[t-1, legIdx, -1, -1]
        thisHeight = poses[t, legIdx, -1, -1]
        nextHeight = poses[t+1, legIdx, -1, -1]
    
        if thisHeight < lastHeight and thisHeight < nextHeight:
            contactHeights = np.append(contactHeights, thisHeight)

    # Hard-coded constants
    DISCARD   = 2
    EPS       = 0.01
    threshold = -sorted(-contactHeights)[DISCARD] + EPS
    return threshold



def get_current_height(angles, fullAngleNames, legIdx):
    ''' Expect angles in degrees, in order specified by anglesTG '''
    pose = angles_to_pose_names(angles.reshape(-1, len(anglesTG)), fullAngleNames)
    return pose[0, legIdx, -1, -1]



################################################################################
# Main disturbance functions for users
# Returns a dict of leg: disturbance (per-timestep)
################################################################################
class DistType(Enum):
    ZERO             = 0
    SLIPPERY_SURFACE = 1
    UNEVEN_SURFACE   = 2
    BUMP_ON_SURFACE  = 3
    SLOPED_SURFACE   = 4
    MISSING_LEG      = 5
    


def get_zero_dists():
    distDict = {}
    for leg in legs:
        legPos        = int(leg[-1])
        numAngles     = len(anglesCtrl[legPos])
        distDict[leg] = np.zeros(numAngles*2)
    return distDict



def get_dists_slippery(maxVelocity):
    ''' Gives disturbances corresponding to walking on slippery surface
        maxVelocity: maximum velocity induced by slip '''
    distDict = {}
    distAngles = {1: np.array([3]),    # femur-tibia flexion
                  2: np.array([1, 2]), # femur rotation, femur-tibia flexion
                  3: np.array([1])}    # femur-tibia flexion
    for leg in legs:
        legPos        = int(leg[-1])
        numAngles     = len(anglesCtrl[legPos])
        distDict[leg] = np.zeros(numAngles*2)
        distDict[leg][numAngles+distAngles[legPos]] = \
            np.random.uniform(-maxVelocity, maxVelocity, len(distAngles[legPos]))
    return distDict



def get_dists_uneven(maxHt):
    ''' Gives disturbances corresponding to walking on uneven surface with many
        random bumps and pits
        maxHt: maximum vertical height of bumps/pits '''
    distDict = {}
    for leg in legs:
        legPos        = int(leg[-1])
        numAngles     = len(anglesCtrl[legPos])
        height        = np.random.uniform(-maxHt, maxHt)
        distDict[leg] = get_dists_endeffector_moves(height, leg)
    return distDict          



def get_dists_bump_or_pit(height, distLeg):
    ''' Gives disturbances simulating one leg stepping on bump or in a pit
        height  : height of bump (positive) or pit (negative)
        distLeg : which leg steps on the bump/pit '''
    distDict = {}
    for leg in legs:
        legPos        = int(leg[-1])
        numAngles     = len(anglesCtrl[legPos])
        distDict[leg] = np.zeros(numAngles*2)

        if leg == distLeg:
            distDict[leg] = get_dists_endeffector_moves(height, leg)
    return distDict



def get_dists_incline_or_decline(angle): 
    ''' Gives disturbances simulating walking on incline/decline
        angle: angle of incline (negative for decline), degrees'''
    femurLen = all_lengths[0][1] / 1000 # Femur length for L1
    bodyLen  = 3 * femurLen # TODO: this was eyeballed
    height   = bodyLen/2*np.tan(np.radians(angle))

    distDict = {}
    for leg in legs:
        legPos        = int(leg[-1])
        numAngles     = len(anglesCtrl[legPos])

        if legPos == 1:
            distDict[leg] = get_dists_endeffector_moves(height, leg)
        elif legPos == 3:
            distDict[leg] = get_dists_endeffector_moves(-height, leg)
        else:
            distDict[leg] = np.zeros(numAngles*2)
    return distDict



def get_dists_missing_leg(missingLeg):
    ''' Gives disturbances simulating walking with a missing leg '''
    # Which legs get affected by missing leg
    distLegs = {'R1': ['R2', 'L1'],
                'R2': ['R1', 'R3'],
                'R3': ['R2', 'L3'],
                'L1': ['L2', 'R1'],
                'L2': ['L1', 'L3'],
                'L3': ['L2', 'R3']}
    height = all_lengths[0][0] / 1000 # TODO: eyeballed; coxa length for L1
    distDict = {}
    for leg in legs:
        legPos        = int(leg[-1])
        numAngles     = len(anglesCtrl[legPos])
        distDict[leg] = np.zeros(numAngles*2)

        if leg in distLegs[missingLeg]:
            distDict[leg] = get_dists_endeffector_moves(height, leg)
    return distDict
