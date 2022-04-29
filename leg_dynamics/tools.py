#!/usr/bin/env python

import copy
import numpy as np
from scipy.misc import derivative

# Use version 0.7.5
# python3 -m pip install sympy==0.7.5
import sympy

# Install from https://github.com/cdsousa/SymPyBotics
# Some customizations required
import sympybotics



def get_numerical_values(subDict, sbCode):
    '''
    subDict: dictionary with robot parameters and states to be subbed
    sbCode : code object from SympyBotics, e.g. rbt.M_code, rbt.c_code
             sbCode[1] contains list of tuples of expressions
             sbCode[2] contains symbolic matrix/vector

    output : matrix/vector evaluated at the given parameters (e.g. M)

    Note that this is faster than using internal functions to show sbCode in terms of q only
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



def x_to_dict(x, legdef):
    ''' Convert vector or list of states x into dictionary form for sympy substitution '''
    stateDict = {}
    for i in range(legdef.dof):
        stateDict[legdef.q[i]]  = x[i] # Angle
        stateDict[legdef.dq[i]] = x[i+legdef.dof] # Angular velocity
    return stateDict



def F2(x, uEqm, paramsDict, legdef, leg):
    '''
    x          : robot state
    uEqm       : torques
    paramsDict : robotic parameters (mass, lengths)
    legdef     : sympybotics definition of leg
    leg        : sympybotics leg object with calculated dynamics
    
    output     : vector of angular acceleration ('F2' in notes)
    '''
    stateDict = x_to_dict(x, legdef)
    subDict   = {**paramsDict, **stateDict}
        
    M = get_numerical_values(subDict, leg.M_code)
    C = get_numerical_values(subDict, leg.C_code)
    g = get_numerical_values(subDict, leg.g_code)

    MInv = M.inv()
    
    term1 = MInv*uEqm
    term2 = MInv*C*x[leg.dof:,:]
    term3 = MInv*g

    return term1 - term2 - term3



def F2_scalar(val, row, xIdx, xEqm, uEqm, paramsDict, legdef, leg):
    ''' 
    val  : value of variable
    row  : which row of F2 to evaluate
    xIdx : which index of x to use as the variable
    
    output : F2 evaluated at xEqm except with a different value at xIdx
    '''
    
    x       = copy.deepcopy(xEqm)
    x[xIdx] = val
    
    return F2(x, uEqm, paramsDict, legdef, leg)[row]



def loadLinearizedSystem():
    ''' Loads files containing ALin, BLin, uEqm from getLinearizedSystem and returns contents '''
    Afn = './data/ALin.dat'
    Bfn = './data/BLin.dat'
    ufn = './data/uEqm.dat'
    
    ALin = np.load(Afn, allow_pickle=True)
    BLin = np.load(Bfn, allow_pickle=True)
    uEqm  = np.load(ufn, allow_pickle=True)
    
    return (ALin, BLin, uEqm)



def getLinearizedSystem(DHTable, linkLengths, linkMasses, xEqm, saveToFiles=True):
    ''' 
    Given physical properties and DH parameters of robot, return linearized system.
    DHTable     : DH table ordered (alpha, a, d, theta)
    linkLengths : list of link lengths
    linkMasses  : list of link masses
    xEqm        : state operating point
    saveToFiles : whether to save the outputs to files
    
    output      : (A, B, uEqm) system matrices and actuator operating point (equilibrium)
                  each is a numpy matrix
    '''
    xEqm = sympy.Matrix(xEqm) # Convert to sympy    
    
    # Generate dynamics 
    legdef = sympybotics.RobotDef('LegRobot', DHTable, dh_convention='standard')
    legdef.frictionmodel = None        
    leg    = sympybotics.RobotDynCode(legdef, verbose=True)
    
    # Generate equilibrium state dictionary (for subbing into sympy)
    stateEqmDict = x_to_dict(xEqm, legdef)

    # Inertia    
    Le = [None] * leg.dof
    for i in range(leg.dof):
        Le[i] = [0] * 6 # Convention: [L_xx, L_xy, L_xz, L_yy, L_yz, L_zz]
    Le[0][5] = 1.0/3.0 * linkMasses[0] * linkLengths[0] * linkLengths[0] # L1_zz
    Le[2][3] = 1.0/3.0 * linkMasses[2] * linkLengths[2] * linkLengths[2] # L3_yy
    Le[3][5] = 1.0/3.0 * linkMasses[3] * linkLengths[3] * linkLengths[3] # L4_zz
    
    # Generate parameters dictionary (for subbing into sympy)
    paramsDict = {}
    for i in range(leg.dof):
        paramsDict[legdef.m[i]] = linkMasses[i]
        for j in range(len(legdef.Le[i])):
            paramsDict[legdef.Le[i][j]] = Le[i][j]
        for j in range(len(legdef.l[i])):
            paramsDict[legdef.l[i][j]] = 0 # Ignore first moment of inertia
    
    subDict = {**paramsDict, **stateEqmDict} # Combined dictionary
    uEqm    = get_numerical_values(subDict, leg.g_code) # eqm input = g
    
    # Calculate matrices for the systems
    B1   = sympy.zeros(leg.dof, leg.dof)
    B2   = get_numerical_values(subDict, leg.M_code).inv()
    BLin = B1.col_join(B2)

    A11 = sympy.zeros(leg.dof, leg.dof)
    A12 = sympy.eye(leg.dof)
    A1  = A11.row_join(A12)
    
    A2 = sympy.zeros(leg.dof, 2*leg.dof) # Jacobian
    for i in range(leg.dof): # row of F2    
        for j in range(2*leg.dof): # element of x
            print(f'Calculating element {i},{j} of Jacobian')
            A2[i,j] = derivative(F2_scalar, xEqm[j], dx=1e-5, 
                      args=(i, j, xEqm, uEqm, paramsDict, legdef, leg))
    ALin = A1.col_join(A2)

    # Convert to numpy
    ALin = np.array(ALin).astype(np.float64)
    BLin = np.array(BLin).astype(np.float64)
    uEqm = np.array(uEqm).astype(np.float64)
    
    # Store to files (since A2 calculation takes a while)
    Afn = './data/ALin.dat'
    Bfn = './data/BLin.dat'
    ufn = './data/uEqm.dat'
    
    ALin.dump(Afn)
    BLin.dump(Bfn)
    uEqm.dump(ufn)
    
    return (ALin, BLin, uEqm)
