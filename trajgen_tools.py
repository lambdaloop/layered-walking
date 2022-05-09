import pickle
from collections import Counter

from angle_functions import *
from model_functions import MLPScaledXY



# Current angles and orders as defined by TG and controller
anglesTG   = ['L1C_flex', 'L1A_rot', 'L1A_abduct', 'L1B_flex', 'L1B_rot']
anglesCtrl = ['L1A_abduct', 'L1B_flex', 'L1B_rot', 'L1C_flex']
mapping    = [2, 3, 4, 0] # tg to ctrl


def tg_to_ctrl(angles):
    ''' Convert angles for use by TG to angles for use by controller '''
    ctrlAngles = np.array([angles[2], angles[3], angles[4], angles[0]])
    return np.radians(ctrlAngles)



def ctrl_to_tg(angles, L1ArotVal):
    ''' 
    Convert angles for use by controller to angles for use by TG
    L1ArotVal isn't currently in the ctrl model
    ''' 
    tgAngles = np.degrees(np.array([angles[3], angles[0], angles[1], angles[2]]))
    return np.append(np.append(tgAngles[0], L1ArotVal), tgAngles[1:])



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
        
        self._angReal = xy_w[0][cc, :self._numAng]
        self._drvReal = xy_w[0][cc, self._numAng:self._numAng*2]
        
        rcos, rsin = xy_w[0][:, [-2, -1]][cc].T
        self._phaseReal = np.arctan2(rsin, rcos)
        self._context   = xy_w[0][cc, -4:-2]
    
    
    def get_initial_vals(self):
        return self._angReal[0], self._drvReal[0], self._phaseReal[0]
    
    
    def step_forward(self, ang, drv, phase, context):        
        inp = np.hstack([ang, drv, context, np.cos(phase), np.sin(phase)])
        out = self._model(inp[None].astype('float32'))[0].numpy()
        ang1, drv1, phase1 = update_state(ang, drv, phase, out, ratio=0.5)
        new_inp = np.hstack([ang1, drv1, context, np.cos(phase1), np.sin(phase1)])
        out = self._model(new_inp[None].astype('float32'))[0].numpy()
        ang, drv, phase = update_state(ang, drv, phase, out, ratio=1.0)
        return ang, drv, phase
