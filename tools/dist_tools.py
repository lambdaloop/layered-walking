#!/usr/bin/env python

import numpy as np

from scipy.optimize import fsolve
from tools.angle_functions import all_lengths, anglesCtrl, get_avg_angles, get_leg_lengths, legs


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
# Main functions for users
# All return a dictionary of disturbances per leg
################################################################################
def get_zero_dists(numSteps):
    distDict = {}
    for leg in legs:
        legPos        = int(leg[-1])
        numAngles     = len(anglesCtrl[legPos])
        distDict[leg] = np.zeros([numAngles*2, numSteps])
    return distDict


def get_dists_slippery(maxVelocity, numSteps):
    ''' Gives disturbances corresponding to walking on slippery surface
        maxVelocity: maximum velocity induced by slip '''
    distDict = {}
    distAngles = {1: np.array([3]),    # femur-tibia flexion
                  2: np.array([1, 2]), # femur rotation, femur-tibia flexion
                  3: np.array([1])}    # femur-tibia flexion
    
    for leg in legs:
        legPos        = int(leg[-1])
        numAngles     = len(anglesCtrl[legPos])
        distDict[leg] = np.zeros([numAngles*2, numSteps])
        distDict[leg][numAngles+distAngles[legPos],:] = \
            np.random.uniform(-maxVelocity, maxVelocity, 
                              [len(distAngles[legPos]), numSteps])
    return distDict



def get_dists_uneven(maxHt, numSteps):
    ''' Gives disturbances corresponding to walking on uneven surface with many
        random bumps and pits
        maxHt: maximum vertical height of bumps/pits '''
    distDict = {}
    for leg in legs:
        legPos        = int(leg[-1])
        numAngles     = len(anglesCtrl[legPos])
        distDict[leg] = np.zeros([numAngles*2, numSteps])
        
        for t in range(numSteps): # TODO: This is hacky slow way
            height = np.random.uniform(-maxHt, maxHt)
            distDict[leg][:,t] = get_dists_endeffector_moves(height, leg)
    return distDict          



def get_dists_bump_or_pit(height, distLeg, numSteps, start, stop):
    ''' Gives disturbances simulating one leg stepping on bump or in a pit
        height  : height of bump (positive) or pit (negative)
        distLeg : which leg steps on the bump/pit '''
    distDict = {}
    for leg in legs:
        legPos        = int(leg[-1])
        numAngles     = len(anglesCtrl[legPos])
        distDict[leg] = np.zeros([numAngles*2, numSteps])

        if leg == distLeg:
            dist = get_dists_endeffector_moves(height, leg)
            for t in range(start, stop):
                distDict[leg][:,t] = dist
    return distDict



def get_dists_incline_or_decline(angle, numSteps): 
    ''' Gives disturbances simulating walking on incline/decline
        angle: angle of incline (negative for decline), degrees'''
    femurLen = all_lengths[0][1] / 1000 # Femur length for L1
    bodyLen  = 3 * femurLen # TODO: this was eyeballed
    height   = bodyLen/2*np.tan(np.radians(angle))

    distDict = {}
    for leg in legs:
        legPos        = int(leg[-1])
        numAngles     = len(anglesCtrl[legPos])
        distDict[leg] = np.zeros([numAngles*2, numSteps])

        if legPos == 1:
            dist = get_dists_endeffector_moves(height, leg)
            for t in range(numSteps):
                distDict[leg][:,t] = dist
        elif legPos == 3:
            dist = get_dists_endeffector_moves(-height, leg)
            for t in range(numSteps):
                distDict[leg][:,t] = dist           
    return distDict



def get_dists_missing_leg(missingLeg, numSteps):
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
        distDict[leg] = np.zeros([numAngles*2, numSteps])

        if leg in distLegs[missingLeg]:
            dist = get_dists_endeffector_moves(height, leg)
            for t in range(numSteps):
                distDict[leg][:,t] = dist
    return distDict

