#!/usr/bin/env python

import control
import copy
import numpy as np
from scipy.misc import derivative
from scipy.linalg import block_diag
from tqdm import trange

from tools.angle_functions import *

# Use version 0.7.5
# python3 -m pip install sympy==0.7.5
import sympy

# Install from https://github.com/cdsousa/SymPyBotics
import sympybotics

# Where to store/load linearized models
# Note: need to make sure this directory exists
directory = './linearized_model/'

# Partial names of linearized model files
Aname = 'ALin.dat'
Bname = 'BLin.dat'
xname = 'xEqm.dat'
uname = 'uEqm.dat'



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



def x_to_dict(x, legDef):
    ''' Convert vector or list of states x into dictionary form for sympy substitution '''
    stateDict = {}
    for i in range(legDef.dof):
        stateDict[legDef.q[i]]  = x[i] # Angle
        stateDict[legDef.dq[i]] = x[i+legDef.dof] # Angular velocity
    return stateDict



def F2(x, uEqm, paramsDict, legDef, legObj):
    '''
    x          : robot state
    uEqm       : torques
    paramsDict : robotic parameters (mass, lengths)
    legDef     : sympybotics definition of leg
    legObj     : sympybotics leg object with calculated dynamics
    
    output     : vector of angular acceleration ('F2' in notes)
    '''
    stateDict = x_to_dict(x, legDef)
    subDict   = {**paramsDict, **stateDict}
        
    M = get_numerical_values(subDict, legObj.M_code)
    C = get_numerical_values(subDict, legObj.C_code)
    g = get_numerical_values(subDict, legObj.g_code)

    MInv = M.inv()
    
    term1 = MInv*uEqm
    term2 = MInv*C*x[legObj.dof:,:]
    term3 = MInv*g

    return term1 - term2 - term3



def F2_scalar(val, row, xIdx, xEqm, uEqm, paramsDict, legDef, legObj):
    ''' 
    val  : value of variable
    row  : which row of F2 to evaluate
    xIdx : which index of x to use as the variable
    
    output : F2 evaluated at xEqm except with a different value at xIdx
    '''
    
    x       = copy.deepcopy(xEqm)
    x[xIdx] = val
    
    return F2(x, uEqm, paramsDict, legDef, legObj)[row]



def generate_linear_filenames(leg):
    ''' Generate filenames for the specified leg, e.g. 'L1' '''
    Afn = directory + leg + '_' + Aname
    Bfn = directory + leg + '_' + Bname
    xfn = directory + leg + '_' + xname
    ufn = directory + leg + '_' + uname
    
    return (Afn, Bfn, xfn, ufn)
    


def load_linearized_system(leg):
    ''' 
    Loads files containing ALin, BLin, xEqm, uEqm generated from 
    get_linearized_system() and returns contents for the specified leg, e.g. 'L1'
    '''
    
    (Afn, Bfn, xfn, ufn) = generate_linear_filenames(leg)
    
    ALin = np.load(Afn, allow_pickle=True)
    BLin = np.load(Bfn, allow_pickle=True)
    xEqm  = np.load(xfn, allow_pickle=True)
    uEqm  = np.load(ufn, allow_pickle=True)
    
    return (ALin, BLin, xEqm, uEqm)



def save_linearized_system(ALin, BLin, xEqm, uEqm, leg):
    (Afn, Bfn, xfn, ufn) = generate_linear_filenames(leg)
    
    ALin.dump(Afn)
    BLin.dump(Bfn)
    xEqm.dump(xfn)
    uEqm.dump(ufn)



def get_linearized_system(DHTable, linkMasses, inertias, xEqm, leg):
    ''' 
    Given physical properties and DH parameters of robot, save linearized system.
    DHTable     : DH table ordered (alpha, a, d, theta)
    linkMasses  : list of link masses
    inertias    : inertias (following Le convention of sympybotics)
    xEqm        : state operating point
    leg         : which leg it is, e.g. 'L1'
    
    output      : Saves (A, B, uEqm, xEqm) system matrices and
                  operating point (equilibrium) to data files as numpy
    '''
    xEqm = sympy.Matrix(xEqm) # Convert to sympy    
    
    # Generate dynamics 
    legDef = sympybotics.RobotDef('LegRobot', DHTable, dh_convention='standard')
    legDef.frictionmodel = None        
    legObj = sympybotics.RobotDynCode(legDef, verbose=True)
    dof    = legObj.dof
    
    # Generate equilibrium state dictionary (for subbing into sympy)
    stateEqmDict = x_to_dict(xEqm, legDef)
    
    # Generate parameters dictionary (for subbing into sympy)
    paramsDict = {}
    for i in range(dof):
        paramsDict[legDef.m[i]] = linkMasses[i]
        for j in range(len(legDef.Le[i])):
            paramsDict[legDef.Le[i][j]] = inertias[i][j]
        for j in range(len(legDef.l[i])):
            paramsDict[legDef.l[i][j]] = 0 # Ignore first moment of inertia
    
    subDict = {**paramsDict, **stateEqmDict} # Combined dictionary
    uEqm    = get_numerical_values(subDict, legObj.g_code) # eqm input = g
    
    # Calculate matrices for the systems
    B1   = sympy.zeros(dof, dof)
    B2   = get_numerical_values(subDict, legObj.M_code).inv()
    BLin = B1.col_join(B2)

    A11 = sympy.zeros(dof, dof)
    A12 = sympy.eye(dof)
    A1  = A11.row_join(A12)
    
    A2 = sympy.zeros(dof, 2*dof) # Jacobian
    for i in trange(dof, desc=' Computing Jacobian ', position=0): # row of F2    
        for j in trange(2*dof, desc=f' Row {i}              ', position=1, leave=False): # element of x
            A2[i,j] = derivative(F2_scalar, xEqm[j], dx=1e-5, 
                      args=(i, j, xEqm, uEqm, paramsDict, legDef, legObj))
    ALin = A1.col_join(A2)

    # Convert to numpy
    ALin = np.array(ALin).astype(np.float64)
    BLin = np.array(BLin).astype(np.float64)
    xEqm = np.array(xEqm).astype(np.float64)
    uEqm = np.array(uEqm).astype(np.float64)
    
    # Store to files
    save_linearized_system(ALin, BLin, xEqm, uEqm, leg)
    
    return



def get_delayed_act_system(Areal, Breal, numDelays):
    ''' Augment system to include delayed actuation '''
    if numDelays == 0:
        return (Areal, Breal)
    
    Nx = Breal.shape[0]
    Nu = Breal.shape[1]
    
    A = np.zeros([(numDelays+1)*Nx, (numDelays+1)*Nx])
    A[0:Nx, 0:Nx] = Areal
    A[0:Nx, numDelays*Nx:(numDelays+1)*Nx] = Breal
    for i in range(numDelays-1):
        A[(i+2)*Nx:(i+3)*Nx, (i+1)*Nx:(i+2)*Nx] = np.eye(Nx)
    
    B = np.zeros([Nx*(numDelays+1), Nu])
    B[Nx:2*Nx, :] = np.eye(Nx)

    return (A, B)



class ControlAndDynamics:
    def __init__(self, leg, anglePen, drvPen, inputPen, Ts, numDelays):
        # Assumes we already ran get_linearized_system() for the appropriate leg
        ALin, BLin, self._xEqm, self._uEqm = load_linearized_system(leg)
        self._Ts        = Ts
        self._leg       = leg
        self._legPos    = int(leg[-1])
        self._numDelays = numDelays
        self._Nr        = len(ALin) # Number of 'real' states
        
        # Zeroth order discretization
        self._Ar  = np.eye(self._Nr) + ALin*Ts
        self._Br  = Ts*BLin
        eigsOL    = np.linalg.eig(self._Ar)[0]
        specRadOL = max(np.abs(eigsOL))
        print(f'Open-loop spectral radius (real system): {specRadOL}')

        # Convert to delayed system
        self._Nx = (numDelays+1)*self._Nr
        self._Nu = int(self._Nr / 2)
        self._A, self._B = get_delayed_act_system(self._Ar, self._Br, numDelays)
        eigsDelayOL    = np.linalg.eig(self._A)[0]
        specRadDelayOL = max(np.abs(eigsDelayOL))
        print(f'Open-loop spectral radius (delayed system): {specRadDelayOL}')

        # Sanity check: controllability
        Qc = control.ctrb(self._A, self._B)
        rankCtrb = np.linalg.matrix_rank(Qc)
        if rankCtrb != self._Nx:
            print('Error: System uncontrollable!')

        # LQR matrices
        QAngle = anglePen * np.eye(self._Nu)
        QDrv   = drvPen * np.eye(self._Nu)
        Q1     = block_diag(QAngle, QDrv)
        Q      = np.zeros([self._Nx, self._Nx])
        Q[0:self._Nx, 0:self._Nx] = Q1 # Only penalize actual states       
        R      = inputPen * np.eye(self._Nu)
        
        # Generate controller
        self._K   = control.dlqr(self._A, self._B, Q, R)[0]
        ACL       = self._A - self._B @ self._K
        eigsCL    = np.linalg.eig(ACL)[0]
        specRadCL = max(np.abs(eigsCL))
        print(f'Closed-loop spectral radius (delayed system): {specRadCL}')
        
    def step_forward(self, yNow, angleNow, angleNxt, drvNow, drvNxt, dist):
        ''' 
        yNow  : includes delay states as well
        dists : size of real system (not including delay states)
        '''   
        # Assumes angles and drv are in TG format
        # dists are disturbances, in ctrl format
        xEqmFlat = self._xEqm.flatten()        
        trajNow  = np.append(tg_to_ctrl(angleNow, self._legPos), 
                             tg_to_ctrl(drvNow/self._Ts, self._legPos))
        trajNxt  = np.append(tg_to_ctrl(angleNxt, self._legPos), 
                             tg_to_ctrl(drvNxt/self._Ts, self._legPos))
        wTraj    = self._A @ (trajNow - xEqmFlat) + xEqmFlat - trajNxt
        
        zeroPadding = np.zeros(self._Nx - self._Nr) # For delay states
        wTrajFull   = np.append(wTraj, zeroPadding)
        distFull    = np.append(dist, zeroPadding)
        
        # Give some look-ahead to wtraj
        uNow = -self._K @ (yNow + wTrajFull)
        yNxt = self._A @ yNow + self._B @ uNow + wTrajFull + distFull
        
        return (uNow, yNxt)


