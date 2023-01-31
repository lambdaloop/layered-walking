#!/usr/bin/env python

import numpy as np
from scipy.linalg import block_diag

np.set_printoptions(suppress=True, linewidth=100000)

n = 3
m = 2
p = 3

dSense = 0
dAct   = 0

AReal = 1*np.ones([n,n])
BReal = 2*np.ones([n,m])
CReal = 3*np.ones([p,n])

# Function starts
#if dSense == 0 and dAct == 0:
# Return original A, B, C    


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


# TODO: build full function and test again
# TODO: update control and dynamics, including dist mtx
# TODO: test in 2- and 3-layer
