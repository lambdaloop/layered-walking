#!/usr/bin/env python

# Use version 0.7.5
# python3 -m pip install sympy==0.7.5
import sympy

# Install from https://github.com/cdsousa/SymPyBotics
import sympybotics

# DH parameter order: (alpha, a, d, theta)

# This is a robot we have dynamics for; sanity check using this one
rbtdef = sympybotics.RobotDef('ReferenceRobot',
                              [('-pi/2', 0, 'q1',   0),
                               (      0, 1,    0,'q2')],
                              dh_convention='standard')
rbtdef.frictionmodel = None

legdef = sympybotics.RobotDef('LegRobot',
                              [(      0, L1,  0, 'q1'),
                               ('-pi/2',  0,  0, 'q2'),
                               ( 'pi/2',  0, L3, 'q3'),
                               (      0, L4,  0, 'q4')],
                              dh_convention='standard')
legdef.frictionmodel = None


# Generate dynamics
rbt = sympybotics.RobotDynCode(rbtdef, verbose=True)

# M(q)*qdotdot + C(q,qdot)qdot + g(q) = tau
# (Ignores non-rotational forces)

# Inertia matrix,  symbolic: rbt.dyn.M, formatted: rbt.M_code
# Coriolis matrix, symbolic: rbt.dyn.C, formatted: rbt.C_code
# Gravity term,    symbolic: rbt.dyn.g, formatted: rbt.g_code
 
