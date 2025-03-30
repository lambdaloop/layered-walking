#!/usr/bin/env python

# only 1 thread, to help parallelize across data
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress warnings from tensorflow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # compute on cpu, it's actually faster for inference with smaller model

# only 1 thread for tf as well
import tensorflow as tf
tf.config.set_soft_device_placement(True)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

import math
import numpy as np
import sys

from tools.ctrl_tools import ControlAndDynamics
from tools.trajgen_tools import TrajectoryGenerator, WalkingData
from tools.angle_functions import legs, \
                            offsets, alphas, kuramato_deriv, \
                            angles_to_pose_names, make_fly_video, \
                            ctrl_to_tg
from tools.dist_tools import *

from tqdm import tqdm, trange
from collections import defaultdict
import os
import pickle
import gc
import time



################################################################################
# User-defined parameters
################################################################################
# filename = '/home/lisa/Downloads/walk_sls_legs_11.pickle'
# filename = '/home/pierre/data/tuthill/models/models_sls/walk_sls_legs_13.pickle'
filename = '/home/lili/data/tuthill/models/models_sls/walk_sls_legs_subang_6.pickle'
# filename = '/home/pierre/models_sls/walk_sls_legs_subang_6.pickle'


numTGSteps     = 900   # How many timesteps to run TG for
Ts             = 1/300 # How fast TG runs
ctrlSpeedRatio = 2     # Controller will run at Ts / ctrlSpeedRatio
ctrlCommRatio  = 8     # Controller communicates to TG this often (as multiple of Ts)

# LQR penalties
drvPen = {'L1': 1e-5, #
          'L2': 1e-5, #
          'L3': 1e-5, # 
          'R1': 1e-5, # 
          'R2': 1e-5, # 
          'R3': 1e-5  #
         }

futurePenRatio = 1.0 # y_hat(t+1) is penalized (ratio)*pen as much as y(t)
                     # y_hat(t+2) is penalized (ratio^2)*pen as much as y(t)
anglePen       = 1e0
inputPen       = 1e-8


numSimSteps = numTGSteps*ctrlSpeedRatio


wd       = WalkingData(filename)

nLegs   = len(legs)
dofTG   = 5


TG      = [None for i in range(nLegs)]
namesTG = [None for i in range(nLegs)]
for ln, leg in enumerate(legs):
    TG[ln] = TrajectoryGenerator(filename, leg, numTGSteps)
    namesTG[ln] = [x[2:] for x in TG[ln]._angle_names]


# bad_delays = [ (10, 2), (10, 16), (20, 2), (20, 8), (35, 2), (35, 4), (40, 2),
#                (45, 2), (45, 6), (50, 2), (55, 2), (60, 2) ]

# bad_delays = [
#     (50, 30),
#     (50, 32.5),
#     (47.5, 2.5),
#     (47.5, 5),
#     (42.5, 2.5),
#     (37.5, 2.5),
#     (35, 2.5),
#     (20, 40),
#     (17.5, 2.5),
#     (12.5, 2.5),
#     (10, 2.5),
#     (10, 17.5),
#     (7.5, 5.5),
#     (2.5, 5),
#     (2.5, 15),
# ]

bad_delays = [
    (37.5, 3.5),
    (12.5, 3.5),
    (10, 3.5),
]

CD_dict = dict()
for act, sense in bad_delays:
    # print(act, sense)
    dAct = int((act / 1000.0) / Ts * ctrlSpeedRatio)
    dSense = int((sense / 1000.0) / Ts * ctrlSpeedRatio)
    good = True
    for leg in legs:
        try:
            CD_dict[ln] = ControlAndDynamics(leg, Ts/ctrlSpeedRatio, dSense, dAct, namesTG[ln])
        except ValueError:
            print("  ERROR leg={}, senseDelay={}, dSense={}, actDelay={}, dAct={}".format(
                repr(leg), sense, dSense, act, dAct))
            good = False
            # break
    print(act, sense, good)

