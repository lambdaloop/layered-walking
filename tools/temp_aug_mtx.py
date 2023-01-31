#!/usr/bin/env python

import numpy as np
from scipy.linalg import block_diag

n = 3
m = 2
p = 3

dSense = 2
dAct   = 3

AReal = 1*np.ones([n,n])
BReal = 2*np.ones([n,m])
CReal = 3*np.ones([p,n])

# Function starts
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
    
A  = np.zeros([nx, nx])

A11  = np.zeros([n1, n1])
A11h = np.eye(n)
for i in range(dAct+1):
    A11h = AReal @ A11h
    A11[n*i:n*(i+1), 0:n] = A11h

A12  = np.zeros([n1, n2])
A12h = BReal
for i in range(dAct+1):
    for j in range(dAct):
        row = (i+j)*n
        if row + n <= r1:
            col = (dAct-j-1)*m
            A12[row:row+n, col:col+m] = A12h
    A12h = AReal @ A12h            

A14  = np.zeros([n4, n4])
A14h = np.eye(n)
for i in range(dAct+1):
    aux     = np.empty([n*i, 0])
    blkdiag = aux # Be careful, shallow copy
    for j in range(dAct+1-i):
        blkdiag = block_diag(blkdiag, A14h)
    blkdiag = block_diag(blkdiag, aux.T)
    A14h    = AReal @ A14h
    A14     += blkdiag

aux = np.empty([m, 0])
A22 = block_diag(aux, np.eye(m*(dAct-1)), aux.T)

A31 = np.zeros([n3, n1])
A31[0:p,0:n] = CReal

aux = np.empty([p, 0])
A33 = block_diag(aux, np.eye(p*(dSense-1)), aux.T)

aux = np.empty([0, n])
A44 = block_diag(aux, np.eye(n*dAct), aux.T)
    
    
        
'''    
    # Middle block of A
    aux                 = np.empty([0, Nx])
    A[N1:2*N1, N1:2*N1] = block_diag(aux, np.eye(Nx*numDelays), aux.T)

    # Construct B matrix
    B                 = np.zeros([NA, Nu])
    B[N1-Nx:N1,:]     = BReal
    B[2*N1:2*N1+Nu,:] = np.eye(Nu)
    
    return (A, B)
    
    #def get_augmented_system(AReal, BReal, CReal, dSense, dAct):
    Augment system (AReal, BReal, CReal) to include
       lookahead             (dAct steps)
       muscle/actuator delay (dAct steps)
       sensor delay          (dSense steps) 
      
       Returns augmented system (A, B, C) 
'''

# TODO: build function step by step as main
# TODO: test augmented matrices on random matrices, including zero-dimensions
# TODO: build full function and test again
# TODO: update control and dynamics, including dist mtx
# TODO: test in 2- and 3-layer
