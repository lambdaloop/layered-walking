#!/usr/bin/env python3

from scipy.spatial.transform import Rotation
import numpy as np
from tqdm import tqdm


def forward_chain(angles, lengths):
    shoulder, flexion, rotation = angles
    cfs = []
    # cc = CoordinateFrame()
    # cfs.append(cc)
    prev_pos = np.array([0, 0, 0])
    prev_rot = Rotation.identity()

    positions = [prev_pos]

    for i in range(5-1):
        if i == 0:
            r = Rotation.from_euler('zyx', [rotation[i], flexion[i], shoulder], degrees=True)
        else:
            r = Rotation.from_euler('zy', [rotation[i], flexion[i]], degrees=True)
        p_local = r.apply([0, 0, lengths[i]])
        # p_global = cfs[i].point_to_world(p_local)
        p_global = prev_pos + prev_rot.apply(p_local)
        positions.append(p_global)
        prev_rot = prev_rot * r
        prev_pos = p_global
        # cc = CoordinateFrame(pos=p_global, rot=cfs[i].rot * r)
        # cfs.append(cc)
    # xyz = np.array([c.pos for c in cfs])
    xyz = np.array(positions)
    return xyz


# row array is in the form:
# A_abduct, A_flex, B_flex, C_flex, D_flex, A_rot, B_rot, C_rot * 6 legs
# 8 * 6 = 40 angles
def angles_to_pose(row, lengths, offsets):
    out = []
    for legnum in range(6):
        a = legnum * 8
        shoulder = row[a+0]
        flex = np.copy(row[a+1:a+5])
        rot = np.append(row[a+5:a+8], 0)
        flex[2] = -np.abs(flex[2])
        flex[1:] = 180 - flex[1:]
        rot[1:] = 180 + rot[1:]
        angles = (shoulder, flex, rot)
        offset = offsets[legnum]
        xyz = forward_chain(angles, lengths[legnum])
        xyz = xyz + offset
        out.append(xyz)
    return out

legs = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']
angnames = ['A_abduct',
            'A_flex', 'B_flex', 'C_flex', 'D_flex',
            'A_rot', 'B_rot', 'C_rot']
full_names = [l + a for l in legs for a in angnames]


all_offsets = np.array(
    [[-0.00229553, -0.00263531, -0.01142328],
     [ 0.05508151, -0.22643036, -0.00824674],
     [ 0.11753684, -0.41916462, -0.00059735],
     [ 0.26658835, -0.01396654, -0.00912715],
     [ 0.1818452 , -0.27336425, -0.013615  ],
     [ 0.2153209 , -0.44930736,  0.01699216]])

all_lengths = np.array(
    [[0.30644581, 0.45009787, 0.34360362, 0.37485521],
     [0.25102419, 0.5403891 , 0.44383802, 0.47973724],
     [0.22397335, 0.53467438, 0.4755834 , 0.45778341],
     [0.26778266, 0.42753417, 0.32381825, 0.38622043],
     [0.2173799 , 0.55001682, 0.43837038, 0.45235905],
     [0.21181969, 0.54009662, 0.4996887 , 0.55621912]])

# median_angles = np.array([
#     146.00028984,   13.3342075 ,   63.11484802,  -89.15836873,
#     132.20974607, -125.00788515,  162.99637495,  164.38310523,
#     139.97148127,   12.16752213,   68.351691  ,  -64.20562219,
#     129.65110559,  158.18397639,  137.70595813,  168.2623689 ,
#     140.21599708,   -1.31702459,   95.47500362,  -69.5466093 ,
#     137.48444563,  134.90378723, -150.358241  ,  133.33405936,
#     149.34826186,   -1.67490245,   74.17772149,  -92.05427172,
#     131.27194079,  -56.56136329, -160.27251345, -163.28854078,
#     147.82392071,   16.14027503,   89.07194766,  -70.65202863,
#     132.02830697,   12.23439621, -152.29595656, -167.29160458,
#     147.65449332,   18.51018665,   97.92399971,  -72.0768633 ,
#     137.46920876,   41.46686536,  161.52502332,    6.17306305])

median_angles = np.array([
    147.57524634,    9.53282204,   63.00287256,  -87.15672335,
    133.74187539, -127.80602784,  163.5438431 ,  167.18400297,
    139.44154607,   10.35239678,   68.67873988,  -70.0745992 ,
    135.39014229,  158.7493212 ,  138.92430412,  170.01201806,
    141.16478714,   -2.66515012,   95.31057094,  -73.71377988,
    141.3813653 ,  136.28316453, -160.3848984 ,  160.96076101,
    151.12961364,   -1.56095286,   72.54123812,  -89.03826747,
    132.189496  ,  -55.47236742, -161.26064865, -166.07505555,
    147.76554823,   15.64910476,   88.43478129,  -75.25671613,
    136.48215761,   12.63745601, -152.39669326, -170.03017531,
    148.9216638 ,   17.77313398,   96.86609836,  -76.26454122,
    141.80029384,   40.77934321,  163.10740062,  122.12763573])

default_angles = dict(zip(full_names, median_angles))

name_to_index = dict(zip(full_names, range(len(full_names))))

def angles_to_pose_multirow(rows, lengths=None, offsets=None, progress=False):
    if lengths is None:
        lengths = all_lengths
    if offsets is None:
        offsets = all_offsets
    out = []
    itr = rows
    if progress:
        itr = tqdm(rows)
    for row in itr:
        pose = angles_to_pose(row, lengths, offsets)
        out.append(pose)
    return np.array(out)


# project v onto u
def proj(u, v):
    if len(u.shape) >= 2:
        return u * (np.sum(v * u, axis = 1) / np.sum(u * u, axis = 1))[:,None]
    else:
        return u * np.dot(v, u) / np.dot(u, u)

def ortho(u, v):
    """Orthagonalize u with respect to v"""
    return u - proj(v, u)

def normalize(u):
    if len(u.shape) >= 2:
        return u / np.linalg.norm(u, axis = 1)[:, None]
    else:
        return u / np.linalg.norm(u)

def angles_flex(vecs, angle):
    a,b,c = angle
    v1 = normalize(vecs[a] - vecs[b])
    v2 = normalize(vecs[c] - vecs[b])
    ang_rad = np.arccos(np.sum(v1 * v2, axis = 1))
    ang_deg = np.rad2deg(ang_rad)
    return ang_deg

def angles_chain(vecs, chain_list):
    chain = []
    flex_type = []
    for c in chain_list:
        if c[-1] == "/":
            chain.append(c[:-1])
            flex_type.append(-1)
        else:
            chain.append(c)
            flex_type.append(1)

    n_joints = len(chain)
    keypoints = np.array([vecs[c] for c in chain])

    xfs = []
    # cc = Rotation.identity()
    cc = Rotation.from_quat([0,0,0,1])
    xfs.append(cc)

    for i in range(n_joints-1):
        pos = keypoints[i+1]
        z_dir = normalize(pos - keypoints[i])
        if i == n_joints - 2: # pick an arbitrary axis for the last joint
            x_dir = ortho([1, 0, 0], z_dir)
            if np.linalg.norm(x_dir) < 1e-5:
                x_dir = ortho([0, 1, 0], z_dir)
        else:
            x_dir = ortho(keypoints[i+2] - pos, z_dir)
            x_dir *= flex_type[i+1]
        x_dir = normalize(x_dir)
        y_dir = np.cross(z_dir, x_dir)
        M = np.dstack([x_dir, y_dir, z_dir])
        rot = Rotation.from_matrix(M)
        xfs.append(rot)

    angles = []
    for i in range(n_joints-1):
        rot = xfs[i].inv() * xfs[i+1]
        ang = rot.as_euler('zyx', degrees=True)
        if i != 0:
            flex = angles_flex(vecs, chain[i-1:i+2]) * flex_type[i]
            test = ~np.isclose(flex, ang[:,1])
            ang[:,0] += 180*test
            ang[:,1] = test*np.mod(-(ang[:,1]+180), 360) + (1-test)*ang[:,1]
            ang = np.mod(np.array(ang) + 180, 360) - 180
        angles.append(ang)

    outdict = dict()
    for i, (name, ang) in enumerate(zip(chain, angles)):
        outdict[name + "_flex"] = ang[:,1]
        if i != len(angles)-1:
            outdict[name + "_rot"] = ang[:,0]
        if i == 0:
            outdict[name + "_abduct"] = ang[:,2]

    return outdict