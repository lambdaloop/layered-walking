import matplotlib
import numpy as np

from tools.trajgen_tools import WalkingData
from tools.angle_functions import legs, anglesTG, \
                                  angles_to_pose_names, make_fly_video

filename        = '/home/lisa/Downloads/walk_sls_legs_11.pickle'
walkingSettings = [11, 0, 0] # walking, turning, flipping speeds (mm/s)
numTGSteps      = 600   # don't change

wd      = WalkingData(filename)
bout    = wd.get_bout(walkingSettings)
angles  = bout['angles']
matplotlib.use('Agg')

dofTG = len(anglesTG)
angle = np.zeros((6, dofTG, numTGSteps))

for i in range(len(legs)):
    angle[i,:,:] = angles[legs[i]].T

angs = angle.reshape(-1, angle.shape[-1]).T
angNames       = [(leg + ang) for leg in legs for ang in anglesTG]
pose_3d        = angles_to_pose_names(angs, angNames)
make_fly_video(pose_3d, 'vids/multileg_real.mp4')


