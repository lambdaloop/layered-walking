import pickle
from collections import Counter

from angle_functions import *
from model_functions import MLPScaledXY
       


def update_state(ang, drv, phase, out, ratio=1.0):
    accel = out[:len(ang)]
    drv1 = drv + accel * ratio
    ang1 = ang + drv * ratio
    phase1 = phase + out[-1]*ratio
    return ang1, drv1, phase1



class TrajectoryGenerator:
    def __init__(self, filename, leg, numAng, numSimSteps):
        self._filename  = filename
        self._leg       = leg
        self._numAng    = numAng
        
        with open(filename, 'rb') as myfile:
            allmodels = pickle.load(myfile)

        # Walking model
        self._model     = MLPScaledXY.from_full(allmodels[leg]['model_walk'])

        # Set up values from training data         
        xy_w, bnums = allmodels[self._leg]['train'] # xy_w is a real trajectory
        common      = Counter(bnums).most_common(100)
        b, _        = common[50]
        cc          = np.where(b == bnums)[0][:numSimSteps]
        
        ang_c = xy_w[0][cc, :self._numAng]
        ang_s = xy_w[0][cc, self._numAng:self._numAng*2]
        self._angReal = np.degrees(np.arctan2(ang_s, ang_c))
        # self._angReal = xy_w[0][cc, :self._numAng]
        self._drvReal = xy_w[0][cc, self._numAng*2:self._numAng*3]

        rcos, rsin = xy_w[0][:, [-2, -1]][cc].T
        self._phaseReal = np.arctan2(rsin, rcos)
        self._context   = xy_w[0][cc, -5:-2]
    
    
    def get_initial_vals(self):
        return self._angReal[0], self._drvReal[0], self._phaseReal[0]
    
    
    def step_forward(self, ang, drv, phase, context):
        rad = np.radians(ang)
        inp = np.hstack([np.cos(rad), np.sin(rad),
                         drv, context, np.cos(phase), np.sin(phase)])
        out = self._model(inp[None].astype('float32'))[0].numpy()
        ang1, drv1, phase1 = update_state(ang, drv, phase, out, ratio=0.5)
        rad1 = np.radians(ang1)
        new_inp = np.hstack([np.cos(rad1), np.sin(rad1),
                             drv1, context, np.cos(phase1), np.sin(phase1)])
        out = self._model(new_inp[None].astype('float32'))[0].numpy()
        ang, drv, phase = update_state(ang, drv, phase, out, ratio=1.2)
        return ang, drv, phase
