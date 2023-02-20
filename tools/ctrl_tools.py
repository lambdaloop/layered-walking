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



def get_augmented_system(AReal, BReal, CReal, dSense, dAct):
    ''' Augment system (AReal, BReal, CReal) to (A, B, C)
        to include delayed sensing, delayed actuation, and lookahead '''
    
    if dSense == 0 and dAct == 0:
        return (AReal, BReal, CReal)
    
    # Original dimensions: state, input, sensor
    n = BReal.shape[0]
    m = BReal.shape[1]
    p = CReal.shape[0]
        
    # Block dimensions of augmented A
    n1 = n*(dAct+1) # real state + lookahead states
    n2 = m*dAct     # act delay states
    n3 = p*dSense   # sense delay states
    n4 = n*(dAct+1) # trajectory states
        
    nx = n1 + n2 + n3 + n4
    nu = m + n3 # allow internal compensation to delayed sensors

    A11  = np.zeros([n1, n1])
    A11h = np.eye(n)
    A12  = np.zeros([n1, n2])
    A12h = BReal
    A14  = np.zeros([n4, n4])
    A14h = np.eye(n)

    for i in range(dAct+1):
        A11h = AReal @ A11h
        A11[n*i:n*(i+1), 0:n] = A11h

        for j in range(dAct):
            row = (i+j)*n
            if row + n <= n1:
                col = (dAct-j-1)*m
                A12[row:row+n, col:col+m] = A12h
        A12h = AReal @ A12h            

        aux     = np.empty([n*i, 0])
        blkdiag = aux # Be careful, shallow copy
        for j in range(dAct+1-i):
            blkdiag = block_diag(blkdiag, A14h)
        blkdiag = block_diag(blkdiag, aux.T)
        A14h    = AReal @ A14h
        A14     += blkdiag

    aux = np.empty([m, 0])
    A22 = np.empty([n2, n2])
    if dAct > 0:
        A22 = block_diag(aux, np.eye(m*(dAct-1)), aux.T)

    A31 = np.zeros([n3, n1])
    if dSense > 0:
        A31[0:p,0:n] = CReal

    aux = np.empty([p, 0])
    A33 = np.empty([n3, n3])
    if dSense > 0:
        A33 = block_diag(aux, np.eye(p*(dSense-1)), aux.T)

    aux = np.empty([0, n])
    A44 = block_diag(aux, np.eye(n*dAct), aux.T)

    # Put A together row-wise
    A1 = np.concatenate([A11, A12, np.zeros([n1,n3]), A14], axis=1)
    A2 = np.concatenate([np.zeros([n2,n1]), A22, np.zeros([n2,n3+n4])], axis=1)
    A3 = np.concatenate([A31, np.zeros([n3,n2]), A33, np.zeros([n3,n4])], axis=1)
    A4 = np.concatenate([np.zeros([n4,n1+n2+n3]), A44], axis=1)
    A  = np.concatenate([A1, A2, A3, A4]);

    B  = np.zeros([nx, nu])
    B[n*dAct:n1, 0:m] = BReal
    B[n1:n1+m, 0:m]   = np.eye(m)
    B[n1+n2:n1+n2+n3, m:nu] = np.eye(p*dSense)  

    C  = np.zeros([p, nx])
    if dSense > 0: # Access delayed sensor reading
        C[:,n1+n2+p*(dSense-1):n1+n2+n3] = np.eye(p)
    else: # Access sensor directly; no delay
        C[:,0:n] = CReal

    return (A, B, C)



def get_augmented_dist_mtx(AReal, dAct):
    Nx   = AReal.shape[0]
    ANow = np.eye(Nx)
    mtx  = np.eye(Nx)
    for i in range(dAct):
        ANow = AReal @ ANow
        mtx  = np.concatenate((mtx, ANow))
    return mtx        
        


class ControlAndDynamics:
    def __init__(self, leg, Ts, dSense, dAct, namesTG=None, lookaheadSTD=None):
        # Assumes we already ran get_linearized_system() for the appropriate leg
        ALin, BLin, self._xEqm, self._uEqm = load_linearized_system(leg)
        self._Ts        = Ts
        self._leg       = leg
        self._legPos    = int(leg[-1])
        self._dSense    = dSense
        self._dAct      = dAct
        self._Nxr       = ALin.shape[0] # Number of 'real' states
        self._Nur       = BLin.shape[1] # Number of 'real' inputs

        if lookaheadSTD is None:
            self._lookaheadSTD = 0
        else:
            self._lookaheadSTD = lookaheadSTD

        if namesTG is None:
            self._namesTG = [x[2:] for x in ANGLE_NAMES_DEFAULT[leg]]
        else:
            self._namesTG = namesTG

        # Zeroth order discretization
        self._Ar  = np.eye(self._Nxr) + ALin*Ts
        self._Br  = Ts*BLin
        self._Cr  = np.eye(self._Nxr) # Perfect sensing

        # Convert to delayed system
        self._A, self._B, self._C = get_augmented_system(self._Ar, self._Br, self._Cr, dSense, dAct)
        self._Nx = self._B.shape[0]
        self._Nu = self._B.shape[1]
        
        # Only used if no delay
        self._Bi = np.linalg.pinv(self._B)

        self._distMtx = get_augmented_dist_mtx(self._Ar, dAct)
        
        # Get penalty matrices
        Q, R, W, V = self.get_penalty_matrices()
        
        # Generate controller and observer
        self._K    = control.dlqr(self._A, self._B, Q, R)[0]        
        self._L    = control.dlqr(self._A.T, self._C.T, W, V)[0].T
        
        self.print_sanity_check()
     
     
    def get_augmented_dist(self, dist):
        augDist       = np.zeros(self._Nx)
        augDist[0:self._Nxr*(self._dAct+1)] = self._distMtx @ dist
        return augDist
    
    
    def get_penalty_matrices(self):
        ''' Get LQG penalty matrices. Relative weighting is hard-coded '''
        ANGLE_PEN  = 1e0
        DERIV_PEN  = 1e-5
        INPUT_PEN  = 1e-8
        FDBK_PEN   = 1e-8
        DIST_SIZE  = 1e0
        NOISE_SIZE = 1e-8
        
        dAct = self._dAct; dSense = self._dSense
        
        QAngle = ANGLE_PEN * np.eye(self._Nur)
        QDrv   = DERIV_PEN * np.eye(self._Nur)
        QState = block_diag(QAngle, QDrv)
        
        Q = np.zeros([self._Nx, self._Nx])
        Q[0:self._Nxr, 0:self._Nxr] = QState
            
        eps  = 1e-8 # Default value for "no" penalty
        RAct = INPUT_PEN * np.eye(self._Nur)   # penalty on true actuation
        RIfp = FDBK_PEN * np.eye(dSense*self._Nxr) # compensatory feedback; no penalty
        R    = block_diag(RAct, RIfp)
            
        W = np.zeros([self._Nx, self._Nx])
        W[0:self._Nxr*(dAct+1),0:self._Nxr*(dAct+1)] = DIST_SIZE * np.eye(self._Nxr*(dAct+1))
        W[self._Nx-self._Nxr:self._Nx,self._Nx-self._Nxr:self._Nx] = np.eye(self._Nxr)
        V = NOISE_SIZE * np.eye(self._Nxr)
        
        return(Q, R, W, V)


    def print_sanity_check(self):
        eigsOL       = np.linalg.eig(self._Ar)[0]
        eigsAugOL    = np.linalg.eig(self._A)[0]
        specRadOL    = max(np.abs(eigsOL))
        specRadAugOL = max(np.abs(eigsAugOL))

        print(f'Open loop spectral radii (should be the same):')
        print(f'Original : {specRadOL:.3f}')
        print(f'Augmented: {specRadAugOL:.3f}')
        
        ABK        = self._A - self._B @ self._K
        ALC        = self._A - self._L @ self._C
        eigsABK    = np.linalg.eig(ABK)[0]
        eigsALC    = np.linalg.eig(ALC)[0]
        specRadABK = max(np.abs(eigsABK))        
        specRadALC = max(np.abs(eigsALC))
        
        print(f'Closed loop spectral radii (should be less than 1):')
        print(f'Controller: {specRadABK:.3f}')
        print(f'Observer  : {specRadALC:.3f}')


    def step_forward(self, xNow, xEst, anglesAhead, drvsAhead, dist):
        ''' 
        xNow       : includes augmented states as well
        xEst       : internal estimation of xNow
        anglesAhead: angles (TG formatted), dAct to dAct+1 steps ahead 
        drvsAhead  : drvs   (TG formatted), dAct to dAct+1 steps ahead
        dist       : size of real system (not including augmented states)
        ''' 
        xEqmFlat = self._xEqm.flatten()
        
        # Synthesize wTraj for t up to t+dAct
        angles = tg_to_ctrl(anglesAhead, self._legPos, self._namesTG)
        drvs   = tg_to_ctrl(drvsAhead, self._legPos, self._namesTG)/self._Ts
        trajs  = np.concatenate((angles, drvs))
        
        # wTraj(t+dAct)
        wTrajAhead = self._A[0:self._Nxr, 0:self._Nxr] @ (trajs[:,0] - xEqmFlat) + \
                     xEqmFlat - trajs[:,1]
    
        # Add noise
        wTrajAhead += np.random.normal(loc=0, scale=self._lookaheadSTD)

        # Update dynamics + estimate with trajectory tracking
        xNow[self._Nx-self._Nxr:self._Nx] = wTrajAhead
        xEst[self._Nx-self._Nxr:self._Nx] = wTrajAhead
            
        # Controller: calculate input        
        uNow = -self._K @ xEst
        
        # Controller: advance estimator
        y       = self._C @ xNow # Sensor input (possibly delayed)
        xEstNxt = self._A @ xEst + self._B @ uNow + self._L @ (y - self._C @ xEst)
        
        # System: advance dynamics
        augDist = self.get_augmented_dist(dist)
        xNxt    = self._A @ xNow + self._B @ uNow + augDist

        return (uNow, xNxt, xEstNxt)

