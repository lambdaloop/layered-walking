#!/usr/bin/env python

import copy

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


def F2_scalar(val, row, linkNum, xEqm, uEqm, paramsDict, legdef, leg):
    ''' 
    val     : value of variable
    row     : which row of F2 to evaluate
    linkNum : which link number's dq we will use as value
    
    output  : F2 evaluated at xEqm except with a different value of dq at the specified linkNum
    '''
    
    x = copy.deepcopy(xEqm)
    x[linkNum+leg.dof] = val
    
    return F2(x, uEqm, paramsDict, legdef, leg)[row]


