import pickle
from collections import Counter, defaultdict
import numpy as np
from tools.angle_functions import *
from tools.model_functions import MLPScaledXY
       


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
        b, _        = common[0]
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


class WalkingData:
    def __init__(self, filename):
        with open(filename, 'rb') as myfile:
            self.data = pickle.load(myfile)

        self._legs = sorted(self.data.keys())

        xy_ws = dict()
        for leg in self._legs:
            xy_w, bnums = self.data[leg]['train']
            xy_ws[leg] = xy_w
        self.bnums = bnums
        self.xy_ws = xy_ws

        k = xy_ws['L1'][0].shape[1]
        self._numAng = (k - 5) // 3

        self.context = xy_ws['L1'][0][:, -5:-2]

        self.counter = Counter(self.bnums)

        self.bnums_ix = defaultdict(list)
        for ix, bnum in enumerate(self.bnums):
            self.bnums_ix[bnum].append(ix)

        self.bnums_uniq = np.unique(self.bnums)
        self.bout_context = []
        for b in self.bnums_uniq:
            sub = self.context[self.bnums_ix[b]]
            avg = np.mean(sub, axis=0)
            self.bout_context.append(avg)
        self.bout_context = np.vstack(self.bout_context)

    def get_initial_vals(self, context, n=1, offset=0, seed=1234):
        np.random.seed(seed)
        dist = np.linalg.norm(self.context - context, axis=1)
        # a little noise to get more different contexts
        dist = dist + np.random.normal(size=dist.shape)*0.1
        ix = np.argsort(dist)
        subix = ix[offset:offset+n]
        return self._get_subix(subix)

    def _get_minlen_bnums(self, min_bout_length=500):
        bouts = []
        for bout, length in self.counter.most_common():
            if length < min_bout_length:
                break
            bouts.append(bout)
        return bouts

    def _get_subix(self, subix):
        contexts = self.context[subix]
        angs = dict()
        drvs = dict()
        phases = dict()

        for leg in self._legs:
            x = self.xy_ws[leg][0]
            rows = x[subix]
            ang_c = rows[:,:self._numAng]
            ang_s = rows[:,self._numAng:self._numAng*2]
            angs[leg] = np.degrees(np.arctan2(ang_s, ang_c))
            drvs[leg] = rows[:,self._numAng*2:self._numAng*3]
            rcos, rsin = rows[:,[-2, -1]].T
            phases[leg] = np.arctan2(rsin, rcos)

        return {
            'angles': angs,
            'derivatives': drvs,
            'phases': phases,
            'contexts': contexts
        }

    def get_bnum(self, bnum):
        subix = self.bnums_ix[bnum]
        return self._get_subix(subix)

    def get_bout(self, context=None, offset=0, min_bout_length=500, seed=1234):
        np.random.seed(seed)
        s = set(self._get_minlen_bnums(min_bout_length))
        check = np.array([b in s for b in self.bnums_uniq])
        if context is None:
            ix = np.arange(len(self.bout_context[check]))
            np.random.shuffle(ix)
        else:
            dist = np.linalg.norm(self.bout_context[check] - context, axis=1)
            # a little noise to get more different contexts
            dist = dist + np.random.normal(size=dist.shape)*0.1
            ix = np.argsort(dist)
        bnum = self.bnums_uniq[check][ix[offset]]
        return self.get_bnum(bnum)
