#!/usr/bin/env python

import copy
import math

# Use version 0.7.5
# python3 -m pip install sympy==0.7.5
import sympy

# Install from https://github.com/cdsousa/SymPyBotics
# Some customizations required
import sympybotics

# TODO: Update parameters
L1 = 1
L3 = 1
L4 = 1

mCoxa  = 1
mFemur = 1
mTibia = 1

# DH parameter order: (alpha, a, d, theta)
legdef = sympybotics.RobotDef('LegRobot',
                              [(      0, L1,  0, 'q1'),
                               ('-pi/2',  0,  0, 'q2'),
                               ( 'pi/2',  0, L3, 'q3'),
                               (      0, L4,  0, 'q4')],
                              dh_convention='standard')
legdef.frictionmodel = None

# Generate dynamics
leg = sympybotics.RobotDynCode(legdef, verbose=True)

# TODO: Update operating point
# Note: sympy trig functions take radians
qEqm    = [None] * leg.dof
qEqm[0] = math.radians(150) # body-coxa flexion
qEqm[1] = math.radians(60)  # coxa-femur flexion
qEqm[2] = math.radians(165) # coxa-femur rotation
qEqm[3] = math.radians(95)  # femur-tibia flexion

m = [mCoxa, 0, mFemur, mTibia] # Link masses

# Inertia
Le = [None] * leg.dof
for i in range(leg.dof):
    Le[i] = [0] * 6 # Convention: [L_xx, L_xy, L_xz, L_yy, L_yz, L_zz]

Le[0][5] = 1.0/3.0 * mCoxa * L1 * L1  # L1_zz
Le[2][3] = 1.0/3.0 * mFemur * L3 * L3 # L3_yy
Le[3][5] = 1.0/3.0 * mTibia * L4 * L4 # L4_zz

subDict = {}

for i in range(leg.dof):
    subDict[legdef.q[i]]  = qEqm[i]
    subDict[legdef.dq[i]] = 3 # Velocities are zero at equilibrium
    subDict[legdef.m[i]]  = m[i]
    
    for j in range(len(legdef.Le[i])):
        subDict[legdef.Le[i][j]] = Le[i][j]
    for j in range(len(legdef.l[i])):
        subDict[legdef.l[i][j]]  = 0 # First moment of inertia


def getNumericalValues(subDict, sbCode):
    '''
    subDict: dictionary with robot parameters (mass, angles, etc.)
    sbCode : code object from SympyBotics, e.g. rbt.M_code, rbt.c_code
             sbCode[1] contains list of tuples of expressions
             sbCode[2] contains symbolic matrix/vector

    output : matrix/vector evaluated at the given parameters (e.g. M)
    '''    
    mySubDict = copy.deepcopy(subDict)
    
    symList    = [x[0] for x in sbCode[0]]
    subbedList = [x[1] for x in sbCode[0]]
    
    allSubbed = False # No symbolic expressions remain; all are numerical
    while not allSubbed:
        allSubbed = True
        for i in range(len(symList)):
            if len(subbedList[i].free_symbols) > 0: # Expression has symbols
                allSubbed = False
                subbedList[i] = subbedList[i].subs(mySubDict)
                mySubDict[symList[i]] = subbedList[i]
    
    return sbCode[1].subs(mySubDict)

g = getNumericalValues(subDict, leg.g_code)

uEqm = g

# M(q)*qdotdot + C(q,qdot)qdot + g(q) = tau
# (Ignores non-rotational forces)

# Inertia matrix  : M_code
# Coriolis matrix : C_code
# Gravity term,   : g_code

