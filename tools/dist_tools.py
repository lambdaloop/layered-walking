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
def get_current_height(angles, fullAngleNames, legIdx):
    ''' Expect angles in degrees, in order specified by anglesTG '''
    try:
        pose = angles_to_pose_names(angles.reshape(-1, len(anglesTG)), fullAngleNames)
        return pose[0, legIdx, -1, -1]
    except:
        return np.nan



def loc_min_detected(locMinWindow, nonRepeatWindow, lastDetection, heights, t):
    ''' Returns boolean of whether heights[center] is a local minimum '''
    if t > locMinWindow*2:  
        center = t - locMinWindow
        if heights[center] == min(heights[center-locMinWindow:t]):
            if t - lastDetection >= nonRepeatWindow:
                return True
    return False



################################################################################
# All-in-one disturbance function; wrapper for all disturbance functions
################################################################################
class DistType(Enum):
    ZERO             = 0
    SLIPPERY_SURFACE = 1
    UNEVEN_SURFACE   = 2
    BUMP_ON_SURFACE  = 3
    SLOPED_SURFACE   = 4
    MISSING_LEG      = 5



def get_dist(distDict, leg):
    ''' distDict should contain keys distType and other key/values for args
        for the specified disturbance type '''
    distType = distDict['distType']
    if distType == DistType.SLIPPERY_SURFACE:
        return get_dists_slippery(distDict['maxVelocity'])[leg]
    elif distType == DistType.UNEVEN_SURFACE:
        return get_dists_uneven(distDict['maxHt'])[leg]
    elif distType == DistType.BUMP_ON_SURFACE:
        return get_dists_bump_or_pit(distDict['height'], distDict['distLeg'])[leg]
    elif distType == DistType.SLOPED_SURFACE:
        return get_dists_incline_or_decline(distDict['angle'])[leg]
    elif distType == DistType.MISSING_LEG:
        return get_dists_missing_leg(distDict['missingLeg'])[leg]
    else: # Default to zero
        return get_zero_dists()[leg]



################################################################################
# Disturbance functions for different scenarios
# All functions return a dict of leg: disturbance (per-timestep)
################################################################################
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
                  3: np.array([1, 2])} # femur rotation, femur-tibia flexion
    for leg in legs:
        legPos        = int(leg[-1])
        numAngles     = len(anglesCtrl[legPos])
        distDict[leg] = np.zeros(numAngles*2)
        
        # Normal distribution about maxVelocity
        dists = np.random.normal(maxVelocity, maxVelocity/10, len(distAngles[legPos]))
        for i in range(len(dists)):
            # Flip a coin to decide whether perturbation is negative or positive
            if np.random.randint(2) == 0: 
                dists[i] = -dists[i]

        distDict[leg][numAngles+distAngles[legPos]] = dists
    return distDict



def get_dists_uneven(maxHt):
    ''' Gives disturbances corresponding to walking on uneven surface with many
        random bumps and pits
        maxHt: maximum vertical height of bumps/pits '''
    distDict = {}
    for leg in legs:
        legPos        = int(leg[-1])
        numAngles     = len(anglesCtrl[legPos])
        
        height = np.random.normal(maxHt, maxHt/10)
        if np.random.randint(2) == 0: # Flip a coin
            height = -height
        
        #height        = np.random.uniform(-maxHt, maxHt)
        
        
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

