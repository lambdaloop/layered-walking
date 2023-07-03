import pickle
from collections import Counter, defaultdict
import numpy as np
from tools.angle_functions import *
from tools.model_functions import MLPScaledXY

import tensorflow as tf


@tf.function
def update_state(ang, drv, phase, out, ratio=1.0):
    accel = out[:len(ang)]
    drv1 = drv + accel * ratio
    ang1 = ang + drv * ratio
    phase1 = phase + out[-1]*ratio
    return ang1, drv1, phase1

ANGLE_NAMES_DEFAULT = {
    'L1': ['L1C_flex', 'L1A_rot', 'L1A_abduct', 'L1B_flex', 'L1B_rot'],
    'L2': ['L2C_flex', 'L2A_rot', 'L2A_abduct', 'L2B_flex', 'L2B_rot'],
    'L3': ['L3C_flex', 'L3A_rot', 'L3A_abduct', 'L3B_flex', 'L3B_rot'],
    'R1': ['R1C_flex', 'R1A_rot', 'R1A_abduct', 'R1B_flex', 'R1B_rot'],
    'R2': ['R2C_flex', 'R2A_rot', 'R2A_abduct', 'R2B_flex', 'R2B_rot'],
    'R3': ['R3C_flex', 'R3A_rot', 'R3A_abduct', 'R3B_flex', 'R3B_rot']
}


class TrajectoryGenerator:
    def __init__(self, filename, leg, numSimSteps):
        self._filename  = filename
        self._leg       = leg

        with open(filename, 'rb') as myfile:
            allmodels = pickle.load(myfile)

        if 'angle_names' in allmodels[leg]:
            self._angle_names = allmodels[leg]['angle_names']
        else:
            # backcompatible with older models
            self._angle_names = ANGLE_NAMES_DEFAULT[leg]

        self._numAng    = len(self._angle_names)

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
    

    @tf.function
    def step_forward(self, ang, drv, phase, context):
        rad = ang * np.pi/180
        inp = tf.concat([tf.cos(rad), tf.sin(rad),
                         drv, context, [tf.cos(phase)], [tf.sin(phase)]], axis=-1)
        out = self._model(inp[None])[0]
        ang1, drv1, phase1 = update_state(ang, drv, phase, out, ratio=0.5)
        rad1 = ang1 * np.pi/180
        new_inp = tf.concat([tf.cos(rad1), tf.sin(rad1),
                             drv1, context, [tf.cos(phase1)], [tf.sin(phase1)]], axis=-1)
        out = self._model(new_inp[None])[0]
        ang, drv, phase = update_state(ang, drv, phase, out, ratio=1.0)
        return ang, drv, phase


    def get_future_traj_raw(self, t1, t2, ang, drv, phase, contexts):
        numVals = t2 - t1
        angs    = []
        drvs    = []
        phases  = []

        tf_contexts = tf.cast(contexts, 'float32')

        out = self.step_forward(
            tf.cast(ang, 'float32'), tf.cast(drv, 'float32'),
            tf.cast(phase, 'float32'), tf_contexts[t1])
        angs.append(out[0])
        drvs.append(out[1])
        phases.append(out[2])

        for i in range(1, numVals):
            out = self.step_forward(angs[i-1], drvs[i-1], phases[i-1],
                                    tf_contexts[t1+i])
            angs.append(out[0])
            drvs.append(out[1])
            phases.append(out[2])

        return angs, drvs, phases

    def get_future_traj(self, t1, t2, ang, drv, phase, contexts):
        angs, drvs, phases = self.get_future_traj_raw(t1, t2, ang, drv, phase, contexts)
        return np.array(angs).T, np.array(drvs).T, np.array(phases)



class WalkingData:
    def __init__(self, filename):
        with open(filename, 'rb') as myfile:
            self.data = pickle.load(myfile)

        self._legs = sorted(self.data.keys())

        if 'angle_names' in self.data[self._legs[0]]:
            self._angle_names = dict([(leg, self.data[leg]['angle_names']) for leg in self._legs])
        else:
            # backcompatible with older models
            self._angle_names = dict([(leg, ANGLE_NAMES_DEFAULT[leg]) for leg in self._legs])

        xy_ws = dict()
        for leg in self._legs:
            xy_w, bnums = self.data[leg]['train']
            xy_ws[leg] = xy_w
        self.bnums = bnums
        self.xy_ws = xy_ws

        # k = xy_ws['L1'][0].shape[1]
        # self._numAng = (k - 5) // 3

        self._numAng = dict([(leg, len(self._angle_names[leg])) for leg in self._legs])

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
            numAng = self._numAng[leg]
            x = self.xy_ws[leg][0]
            rows = x[subix]
            ang_c = rows[:,:numAng]
            ang_s = rows[:,numAng:numAng*2]
            angs[leg] = np.degrees(np.arctan2(ang_s, ang_c))
            drvs[leg] = rows[:,numAng*2:numAng*3]
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
