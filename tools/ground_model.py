#!/usr/bin/env ipython

from tools.angle_functions import anglesCtrl
from tools.angle_functions import angles_chain, forward_chain, default_angles, \
    angnames, leg_lengths, default_positions, run_forward
import numpy as np
from scipy import optimize

def make_optimizer_fun(leg, names, offset=None):
    full = np.array([default_angles[leg + a] for a in angnames])
    sub_ixs = [angnames.index(n) for n in names]
    init = full[sub_ixs]
    lengths = leg_lengths[leg]
    if offset is None:
        offset = np.zeros(3)
    else:
        offset = np.array(offset)
    def get_positions(angles):
        full[sub_ixs] = angles
        xyz = run_forward(full, lengths)
        return xyz + offset
    def optimizer(angles, target):
        xyz = get_positions(angles)
        return np.linalg.norm(xyz[-1] - target)
    return optimizer, init, get_positions

class GroundModel:
    def __init__(self, height, incline=0):
        # get angle definition
        # set up inverse kinematics model
        # it should be across all the legs in order to update velocity as well
        # TODO: take in names from trajgen angles

        self._height = height # measured from L1A
        self._incline = incline # in degrees

        # set up IK models for each leg, based on angle definitions
        legs = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']
        self.optimizers = dict()
        self.get_positions = dict()
        for leg in legs:
            names = anglesCtrl[int(leg[1])]
            opt, _, pos = make_optimizer_fun(leg, names, offset=default_positions[leg + 'A'])
            self.optimizers[leg] = opt
            self.get_positions[leg] = pos

    def step_forward(self, angles_prev, angles_curr, velocities):
        # takes angles as input and outputs new angles, corrected for ground collisions
        # (also velocities)
        # format is dictionary of (leg, angnames)
        # this will go through all the legs present in the angles dictionary
        legs = sorted(angles_curr.keys())

        # convert angles from rad -> deg -> rad
        # unit of velocity is rad/s * sampling_time

        pos_curr = dict()
        pos_prev = dict()
        delta = dict()
        ground_legs = []

        # estimate deltas
        for leg in legs:
            xyz = pos_curr[leg] = self.get_positions[leg](angles_curr[leg])
            pos_prev[leg] = self.get_positions[leg](angles_prev[leg])
            delta[leg] = pos_curr[leg][-1] - pos_prev[leg][-1]
            if xyz[-1, 2] < -1 * self._height:
                ground_legs.append(leg)


        average_delta = np.mean([delta[leg] for leg in ground_legs], axis=0)

        angles_next = dict(angles_curr)
        velocities_next = dict(velocities)
        # apply height and delta correction to ground connected legs
        for leg in ground_legs:
            xyz = pos_curr[leg]
            target = np.copy(xyz[-1])
            target[0] = average_delta[0] + pos_prev[leg][-1, 0]
            target[1] = average_delta[1] + pos_prev[leg][-1, 1]
            target[2] = -self._height
            opt = optimize.least_squares(
                self.optimizers[leg], angles_curr[leg],
                args=(target,))
            angles_next[leg] = opt.x
            velocities_next[leg] = angles_next[leg] - angles_prev[leg] # delta t


        return angles_next, velocities_next

        # for each leg in connection with ground
        # - estimate its delta compared to previous timepoint
        # - average planar delta within the set of legs hitting ground
        # - apply planar delta to get final leg positions

        # run forward kinematics on legs with current angles
        # move leg tips to correct for height, average planar movement of each leg
        # run inverse kinematics to get the appropriate angles
        # correct the velocities for the ground collisions