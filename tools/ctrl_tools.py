#!/usr/bin/env python

import control
import copy
import numpy as np
from scipy.misc import derivative
from scipy.linalg import block_diag
from tqdm import trange

from tools.angle_functions import *
from tools.trajgen_tools import ANGLE_NAMES_DEFAULT

# Use version 0.7.5
# python3 -m pip install sympy==0.7.5
import sympy

import warnings # Suppress warnings from sympybotics
warnings.filterwarnings("ignore", category=SyntaxWarning)

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
    f = get_numerical_values(subDict, legObj.f_code)

    MInv = M.inv()
    return MInv*(uEqm - C*x[legObj.dof:,:] - g - f)
    


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



def get_linearized_system(DHTable, linkMasses, inertias, frics, xEqm, leg):
    ''' 
    Given physical properties and DH parameters of robot, save linearized system.
    DHTable     : DH table ordered (alpha, a, d, theta)
    linkMasses  : list of link masses
    inertias    : inertias (following Le convention of sympybotics)
    frics       : list of friction coefficients
    xEqm        : state operating point
    leg         : which leg it is, e.g. 'L1'
    
    output      : Saves (A, B, uEqm, xEqm) system matrices and
                  operating point (equilibrium) to data files as numpy
    '''
    xEqm = sympy.Matrix(xEqm) # Convert to sympy    
    
    # Generate dynamics 
    legDef = sympybotics.RobotDef('LegRobot', DHTable, dh_convention='standard')
    legDef.frictionmodel = {'viscous'}       
    legObj = sympybotics.RobotDynCode(legDef, verbose=True)
    dof    = legObj.dof
    
    # Generate equilibrium state dictionary (for subbing into sympy)
    stateEqmDict = x_to_dict(xEqm, legDef)
    
    # Generate parameters dictionary (for subbing into sympy)
    paramsDict = {}
    for i in range(dof):
        paramsDict[legDef.m[i]]  = linkMasses[i]
        paramsDict[legDef.fv[i]] = frics[i]
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



def get_augmented_system(AReal, BReal, numDelays):
    ''' Augment system to include delayed actuation and lookahead '''
    if numDelays == 0:
        return (AReal, BReal)
    Nx = BReal.shape[0]
    Nu = BReal.shape[1]
        
    # Dimensions of blocks
    N1 = Nx*(numDelays+1)
    N2 = Nu*numDelays
    NA = 2*N1 + N2
        
    A     = np.zeros([NA, NA])
    ANow1 = np.eye(Nx) # Powers of A for upper left block of A
    ANow2 = np.eye(Nx) # Powers of A for upper middle block of A
    ABNow = BReal      # Powers of A*B for upper right block of A
        
    for i in range(numDelays+1):
        # Upper left block of A
        ANow1 = AReal @ ANow1
        A[Nx*i:Nx*(i+1), 0:Nx] = ANow1
        
        # Upper middle block of A
        aux     = np.empty([Nx*i, 0])
        blkdiag = aux # Be careful, shallow copy
        for j in range(numDelays+1-i):
            blkdiag = block_diag(blkdiag, ANow2)
        blkdiag = block_diag(blkdiag, aux.T)
        ANow2   = AReal @ ANow2
        A[0:N1, N1:2*N1] += blkdiag
        
        # Upper right block of A
        for j in range(numDelays):
            rowStart = (i+j)*Nx
            rowEnd   = rowStart + Nx
            if rowEnd <= N1:
                colStart = 2*N1+(numDelays-j-1)*Nu
                colEnd   = colStart + Nu            
                A[rowStart:rowEnd, colStart:colEnd] = ABNow
        ABNow = AReal @ ABNow            

    # Middle block of A
    aux                 = np.empty([0, Nx])
    A[N1:2*N1, N1:2*N1] = block_diag(aux, np.eye(Nx*numDelays), aux.T)

    # Bottom right block of A
    aux                 = np.empty([Nu, 0])
    A[2*N1:NA, 2*N1:NA] = block_diag(aux, np.eye(Nu*(numDelays-1)), aux.T)

    # Construct B matrix
    B                 = np.zeros([NA, Nu])
    B[N1-Nx:N1,:]     = BReal
    B[2*N1:2*N1+Nu,:] = np.eye(Nu)
    
    return (A, B)



def get_augmented_dist_mtx(AReal, numDelays):
    Nx   = AReal.shape[0]
    ANow = np.eye(Nx)
    mtx  = np.eye(Nx)
    for i in range(numDelays):
        ANow = AReal @ ANow
        mtx  = np.concatenate((mtx, ANow))
    return mtx       



class ControlAndDynamics:
    def __init__(self, leg, Ts, numDelays, futurePenRatio, anglePen, drvPen, inputPen, namesTG=None):
        # Assumes we already ran get_linearized_system() for the appropriate leg
        ALin, BLin, self._xEqm, self._uEqm = load_linearized_system(leg)
        self._Ts        = Ts
        self._leg       = leg
        self._legPos    = int(leg[-1])
        self._numDelays = numDelays
        self._Nr        = ALin.shape[0] # Number of 'real' states

        if namesTG is None:
            self._namesTG = [x[2:] for x in ANGLE_NAMES_DEFAULT[leg]]
        else:
            self._namesTG = namesTG

        # Zeroth order discretization
        self._Ar  = np.eye(self._Nr) + ALin*Ts
        self._Br  = Ts*BLin
        eigsOL    = np.linalg.eig(self._Ar)[0]
        specRadOL = max(np.abs(eigsOL))
        # print(f'Open-loop spectral radius (real system): {specRadOL:.3f}')

        # Convert to delayed system
        self._A, self._B = get_augmented_system(self._Ar, self._Br, numDelays)
        self._Nx = self._B.shape[0]
        self._Nu = self._B.shape[1]
        eigsDelayOL    = np.linalg.eig(self._A)[0]
        specRadDelayOL = max(np.abs(eigsDelayOL))
        # print(f'Open-loop spectral radius (delayed system): {specRadDelayOL:.3f}')

        # Only used if no delay
        self._Bi = np.linalg.pinv(self._B)

        self._distMtx = get_augmented_dist_mtx(self._Ar, numDelays)
        
        # State and input penalty matrices        
        QAngle = anglePen * np.eye(self._Nu)
        QDrv   = drvPen * np.eye(self._Nu)
        QState = block_diag(QAngle, QDrv)
        
        Q = np.zeros([self._Nx, self._Nx])
        for i in range(numDelays+1): # Penalize state and predicted states
            start = i*self._Nr
            end   = (i+1)*self._Nr
            Q[start:end, start:end] = QState*pow(futurePenRatio,i)        
        R = inputPen * np.eye(self._Nu)
        
        # Generate controller
        self._K   = control.dlqr(self._A, self._B, Q, R)[0]
        ACL       = self._A - self._B @ self._K
        eigsCL    = np.linalg.eig(ACL)[0]
        specRadCL = max(np.abs(eigsCL))
        # print(f'Closed-loop spectral radius (delayed system): {specRadCL:.3f}')
    
    def get_augmented_dist(self, dist):
        augDist       = np.zeros(self._Nx)
        N1            = self._Nr*(self._numDelays+1)
        augDist[0:N1] = self._distMtx @ dist
        return augDist
    
    def step_forward(self, yNow, anglesAhead, drvsAhead, dist):
        ''' 
        yNow       : includes augmented states as well
        anglesAhead: angles (TG formatted), numDelay to numDelay+1 steps ahead 
        drvsAhead  : drvs   (TG formatted), numDelay to numDelay+1 steps ahead
        dist       : size of real system (not including augmented states)
        ''' 
        N1       = self._Nr*(self._numDelays+1)
        xEqmFlat = self._xEqm.flatten()
        
        # Synthesize wtraj for t up to t+numDelay
        angles = tg_to_ctrl(anglesAhead, self._legPos, self._namesTG)
        drvs   = tg_to_ctrl(drvsAhead, self._legPos, self._namesTG)/self._Ts
        trajs  = np.concatenate((angles, drvs))
        
        # wTraj(t+numDelay)
        wTrajAhead = self._A[0:self._Nr, 0:self._Nr] @ (trajs[:,0] - xEqmFlat) + \
                     xEqmFlat - trajs[:,1]
        
        # Set yNow's wtraj(t+numDelay) state appropriately
        if self._numDelays > 0:
            yNow[2*N1-self._Nr:2*N1] = wTrajAhead
        
        # Calculate input
        uNow = -self._K @ yNow
        if self._numDelays == 0:
            uNow -= self._Bi @ wTrajAhead
        
        # Advance dynamics
        augDist = self.get_augmented_dist(dist)
        yNxt    = self._A @ yNow + self._B @ uNow + augDist
        if self._numDelays == 0:
            yNxt += wTrajAhead

        return (uNow, yNxt)


