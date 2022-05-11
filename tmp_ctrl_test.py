import control
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import block_diag

from ctrl_tools import *
from trajgen_tools import *


Ts = 1/300 # Sampling time

# Assumes we already saved the linearized system in the appropriate files
# by running getLinearizedSystem()
ALin, BLin, xEqm, uEqm = loadLinearizedSystem()

dof = int(len(ALin)/2)
Nx  = 2*dof
Nu  = dof

# Discretize using e^(A*T) ~= I + A*T
A = np.eye(Nx) + ALin*Ts
B = Ts*BLin + 0.5*Ts*Ts*ALin @ BLin
eigsOL    = np.linalg.eig(A)[0]
specRadOL = max(np.abs(eigsOL))
print(f'Open-loop spectral radius: {specRadOL}')

# Sanity check: controllability
Qc = control.ctrb(A,B)
rankCtrb = np.linalg.matrix_rank(Qc)
if rankCtrb != Nx:
    print('Error: System uncontrollable!')

# Controller objective
anglePenalty    = 1e-0
velocityPenalty = 1e-0 # Making this too small causes issues
inputPenalty    = 1e-8

Q1 = anglePenalty*np.eye(dof)
Q2 = velocityPenalty*np.eye(dof)
Q  = block_diag(Q1, Q2)
R  = inputPenalty*np.eye(Nu)

K   = control.dlqr(A, B, Q, R)[0]
ACL = A - B @ K
eigsCL    = np.linalg.eig(ACL)[0]
specRadCL = max(np.abs(eigsCL))
print(f'Closed-loop spectral radius: {specRadCL}')

tSim = 200

xs = np.zeros((Nx, tSim))
xs[3, 0] = 1
us = np.zeros((Nu, tSim))

for t in range(tSim-1):
    us[:,t]   = -K @ xs[:,t]
    xs[:,t+1] = A @ xs[:,t] + B @ us[:,t]

plt.figure(1)
plt.clf()
for i in range(Nx):
    plt.subplot(2,5,i+1)
    plt.title(anglesCtrl[i%dof])
    plt.plot(xs[i,:])    
plt.show()







