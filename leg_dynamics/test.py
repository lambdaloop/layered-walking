#!/usr/bin/env python

# Use version 0.7.5
# python3 -m pip install sympy==0.7.5
import sympy

# Install from https://github.com/cdsousa/SymPyBotics
# Some customizations required
import sympybotics

# Parameters
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

# Custom definition of inertia
Le = [None] * 4
for i in range(len(Le)):
    Le[i] = [0] * 6 
    # Convention: [L_xx, L_xy, L_xz, L_yy, L_yz, L_zz]

Le[0][5] = 1.0/3.0 * mCoxa * L1 * L1  # L1_zz
Le[2][3] = 1.0/3.0 * mFemur * L3 * L3 # L3_yy
Le[3][5] = 1.0/3.0 * mTibia * L4 * L4 # L4_zz

legdef.set_Le(Le)
legdef.set_l_zero() # Ignore first moment of inertia

# Generate dynamics
leg = sympybotics.RobotDynCode(legdef, verbose=True)


# M(q)*qdotdot + C(q,qdot)qdot + g(q) = tau
# (Ignores non-rotational forces)

# Inertia matrix,  symbolic: dyn.M, formatted: M_code
# Coriolis matrix, symbolic: dyn.C, formatted: C_code
# Gravity term,    symbolic: dyn.g, formatted: g_code

