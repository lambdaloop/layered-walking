#!/usr/bin/env python

import math

# Use version 0.7.5
# python3 -m pip install sympy==0.7.5
import sympy

# Install from https://github.com/cdsousa/SymPyBotics
# Some customizations required
import sympybotics

from tools import *

#############################################################
# User defined parameters
#############################################################
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
#############################################################

# Generate dynamics
leg = sympybotics.RobotDynCode(legdef, verbose=True)


# TODO: Update operating point
# Note: sympy trig functions take radians
xEqm    = sympy.zeros(leg.dof * 2, 1)
xEqm[0] = math.radians(150) # body-coxa flexion
xEqm[1] = math.radians(60)  # coxa-femur flexion
xEqm[2] = math.radians(165) # coxa-femur rotation
xEqm[3] = math.radians(95)  # femur-tibia flexion

stateEqmDict = x_to_dict(xEqm, legdef)


# Generate parameters dictionary
m = [mCoxa, 0, mFemur, mTibia] # Link masses

# Inertia
Le = [None] * leg.dof
for i in range(leg.dof):
    Le[i] = [0] * 6 # Convention: [L_xx, L_xy, L_xz, L_yy, L_yz, L_zz]

Le[0][5] = 1.0/3.0 * mCoxa * L1 * L1  # L1_zz
Le[2][3] = 1.0/3.0 * mFemur * L3 * L3 # L3_yy
Le[3][5] = 1.0/3.0 * mTibia * L4 * L4 # L4_zz

paramsDict   = {} # Robotic parameters: mass, lengths, inertia
for i in range(leg.dof):
    paramsDict[legdef.m[i]]  = m[i]   
    for j in range(len(legdef.Le[i])):
        paramsDict[legdef.Le[i][j]] = Le[i][j]
    for j in range(len(legdef.l[i])):
        paramsDict[legdef.l[i][j]]  = 0 # Ignore first moment of inertia

subDict = {**paramsDict, **stateEqmDict}

g    = get_numerical_values(subDict, leg.g_code)
uEqm = g

# TODO: new development

from scipy.misc import derivative

# These are zero indexed
row     = 1
linkNum = 2
a = derivative(F2_scalar, xEqm[leg.dof+linkNum], dx=1e-5, args = (row, linkNum, xEqm, uEqm, paramsDict, legdef, leg))

print(a)


