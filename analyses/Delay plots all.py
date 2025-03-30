#!/usr/bin/env python
# coding: utf-8

# Perturbation plots
# ==================
# 
# 

# ## Goal
# 
# 

# The goal of this notebook is to make plots for the delays figure of the paper. Namely:
# 
# -   Example perturbed trajectories with different delays
# -   Effect of delta perturbation on other legs
# -   Heatmap of magnitude of slippery perturbation vs delays
# 
# The network should be set up so that running it from top to bottom makes all the figure sublayouts.
# 
# 

# ## Setup
# 
# 

# In[1]:


import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt
import os
from tqdm import tqdm, trange
import pandas as pd
from scipy import signal
from scipy.stats import gaussian_kde
from glob import glob
import pickle


# In[2]:


import sys
project_path = "/home/lili/research/tuthill/layered-walking"
data_path = '/home/lili/data/tuthill/models/sls_runs'
sys.path.append(project_path)
from tools.angle_functions import anglesTG as angle_names_1leg
from tools.angle_functions import legs
from tools.angle_functions import make_fly_video, angles_to_pose_names
from tools.trajgen_tools import WalkingData
from tools.dist_tools import DistType


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('default')
plt.rcParams['figure.figsize'] = (7.5, 2)


# In[4]:


xvals = np.linspace(-np.pi, np.pi)
def get_phase(ang):
    m = np.median(ang, axis=0)
    s = np.std(ang, axis=0)
    s[s == 0] = 1
    dm = (ang - m) / s
    phase = np.arctan2(-dm[:,1], dm[:,0])
    return phase

def phase_align_poly(ang, extra=None, order=5):
    topredict = ang
    if extra is not None:
        topredict = np.hstack([ang, extra])
    means = np.full((len(xvals), topredict.shape[1]), np.nan)
    stds = np.full((len(xvals), topredict.shape[1]), np.nan)
    if len(ang) < 50: # not enough data
        return means, stds
    phase = get_phase(ang)
    # pcs = np.vstack([np.cos(phase), np.sin(phase)]).T
    b = np.vstack([np.cos(i * phase + j) for i in range(order) for j in [0, np.pi/2]]).T
    xcs = np.vstack([np.cos(i * xvals + j) for i in range(order) for j in [0, np.pi/2]]).T
    for i in range(topredict.shape[1]):
        cc = np.isfinite(topredict[:,i])
        model = sm.OLS(topredict[cc,i], b[cc]).fit()
        means[:,i] = model.predict(xcs)
        s, upper,lower = wls_prediction_std(model, xcs)
        stds[:,i] = s
    return means, stds


# In[5]:


# list(d.keys())


# In[6]:


fname_pat = os.path.join(data_path, 'delays_stats_subang_v9_actuate_*.pkl')
fnames = sorted(glob(fname_pat))

fname_pat = os.path.join(data_path, 'delays_stats_subang_v9_sense_poisson*.pkl')
fnames += sorted(glob(fname_pat))

fname_pat = os.path.join(data_path, 'delays_stats_subang_v9_sense_gaussian*.pkl')
fnames += sorted(glob(fname_pat))

conditions = []
angles = []
derivs = []
accels = []

for fname in tqdm(fnames, ncols=70):
    # d = np.load(fname, allow_pickle=True)
    with open(fname, 'rb') as f:
        d = pickle.load(f)

    angle_names = d['angleNames'][0]

    for i, cond in enumerate(d['conditions']):
        ang = d['angle'][i]
        deriv = signal.savgol_filter(ang, 5, 2, deriv=1, axis=0)
        accel = signal.savgol_filter(deriv, 5, 2, deriv=1, axis=0)
        if 'senseDelay' not in cond:
            cond['senseDelay'] = 0.010 # fixed
        if 'actDelay' not in cond:
            cond['actDelay'] = 0.030
        conditions.append(cond)
        angles.append(ang)
        derivs.append(deriv)
        accels.append(accel)


# In[7]:


conditions[0]


# In[8]:


perturb_ranges = {
    'before': (0, 300),
    'during': (300, 600),
    'after': (600, 900)
}


# In[9]:


speeds = np.array([x['context'] for x in conditions])
dist_types = np.array([x['dist'].value for x in conditions])
act_delays = np.array([x['actDelay'] for x in conditions])
sense_delays = np.array([x['senseDelay'] for x in conditions])
max_velocities = np.array([x['maxVelocity'] for x in conditions])
angle_names = list(angle_names)


# In[10]:


fname = '/home/lili/data/tuthill/models/models_sls/data_subang_5.pickle'
wd = WalkingData(fname)


# In[11]:


wd.bout_context


# In[12]:


fancy_angle_names = {
    'C_flex': 'femur-tibia\nflexion',
    'A_rot': 'coxa\nrotation',
    'A_abduct': 'body-coxa\nflexion',
    'B_flex': 'coxa-femur\nflexion',
    'B_rot': 'femur\nrotation'
}


# ## principal component metric
# 
# 

# In[13]:


full_L = []
bnums = wd._get_minlen_bnums(300)
for bnum in bnums:
    bout = wd.get_bnum(bnum)
    angs = np.hstack([bout['angles'][leg] for leg in legs])
    full_L.append(angs)
full = np.vstack(full_L)

full_sc = np.hstack([np.sin(np.deg2rad(full)),
                     np.cos(np.deg2rad(full))])

pca = PCA(n_components=2)
pcs = pca.fit_transform(full_sc)

subixs = np.random.choice(pcs.shape[0], size=10000, replace=False)
kde = gaussian_kde(pcs[subixs,:2].T)


# In[14]:


pca.explained_variance_


# In[15]:


angles_sc = np.dstack([np.sin(np.deg2rad(angles)),
                         np.cos(np.deg2rad(angles))])

angles_sc_flat = angles_sc.reshape(-1, angles_sc.shape[-1])

pcs = np.full((angles_sc_flat.shape[0], 2), np.nan)
good = np.all(np.isfinite(angles_sc_flat), axis=1)
pcs[good] = pca.transform(angles_sc_flat[good])
# pcs = pca.transform(angles_sc_flat)

pdfs_flat = np.full(len(pcs), -2.5)
step = 500
for i in trange(0, len(pcs), step, ncols=70):
  check = np.all(np.isfinite(pcs[i:i+step]), axis=1)
  pdfs_flat[i:i+step][check] = kde.logpdf(pcs[i:i+step, :2][check].T)

pdfs_shaped = pdfs_flat.reshape(angles_sc.shape[:2])


# In[16]:


np.savez_compressed('angle_pdfs.npz', pdfs=pdfs_shaped)


# ## Actuation delay plots
# 
# 

# ### Example time series
# 
# 

# For the figure part A, we'd like to have multiple example angles. Perhaps R1 femur-tibia flexion would be good to show, but also L2 femur rotation?
# I'd like to have multiple traces for each angle, perhaps we could show with forward, rotation, and sideslip?
# 
# In this code, we could also make a supplementary figure with a more complete set of angles.
# 
# 

# In[60]:


# # plot_speeds = [[[[12, 0, 0]]]]
# # plot_delays = [0, 0.015, 0.030, 0.045]
# plot_delays = [0, 0.010, 0.020, 0.030, 0.040]
# # plot_velocities = [8, 14]
# plot_speed = 12


# # In[61]:

## Use delay plots sensory actuation right now

# angnames = ['R1C_flex', 'L2B_rot']
# dists = ['poisson', 'impulse']
# dist_values = {'poisson': DistType.POISSON_GAUSSIAN.value,
#                'impulse': DistType.IMPULSE.value}

# for dist in dists:
#     dist_value = dist_values[dist]
#     if dist == 'poisson':
#         max_velocity_constant = 1.875
#     else:
#         max_velocity_constant = 5.0

#     for angname in angnames:
#         ix_ang = angle_names.index(angname)

#         plt.figure(figsize=(5.5, 2))
#         for i in range(len(plot_delays)):
#             plt.subplot(len(plot_delays), 1, i+1)
#             ixs = np.where((speeds[:, 0] == plot_speed)
#                            & (dist_types == dist_value)
#                            & np.isclose(act_delays, plot_delays[i])
#                            & np.isclose(max_velocities, max_velocity_constant)
#                            & np.isclose(sense_delays, 0.01)
#                            )[0]
#             ix_bout = ixs[3]
#             print(conditions[ix_bout])
#             ang = angles[ix_bout][:, ix_ang]
#             t = np.arange(len(ang))/300.0
#             if angname == 'R1C_flex':
#                 ang = np.clip(ang, 0, 180)
#             elif angname == 'L2B_rot':
#                 ang = np.mod(ang, 360)
#             plt.plot(t, ang)
#             plt.axvline(t[300], color='gray', linestyle='dotted')
#             plt.axvline(t[600], color='gray', linestyle='dotted')
#             if angname == 'R1C_flex':
#                 plt.ylim(0, 180)
#                 plt.yticks([60, 120])
#             else:
#                 plt.ylim(0, 360)
#                 plt.yticks([120, 240])
#             if i != 2:
#                 plt.xticks(ticks=[])

#         sns.despine(bottom=True)
#         plt.ylabel("Angle (deg)")
#         plt.xlabel("Time (s)")

#         plt.savefig('plots/act_delays_trace_{}_{}.pdf'.format(angname, dist),
#                     bbox_inches = "tight")


# ### Videos
# 
# 

# In[62]:


# dists = ['poisson', 'impulse']
# dist_values = {'poisson': DistType.POISSON_GAUSSIAN.value,
#                'impulse': DistType.IMPULSE.value}

# for dist in dists:
#     if dist == 'poisson':
#         max_velocity_constant = 1.875
#     else:
#         max_velocity_constant = 5.0

#     for i in range(len(plot_delays)):
#         dist_value = dist_values[dist]

#         ixs = np.where((speeds[:, 0] == plot_speed)
#                        & (dist_types == dist_value)
#                        & np.isclose(act_delays, plot_delays[i])
#                        & np.isclose(max_velocities, max_velocity_constant)
#                        & np.isclose(sense_delays, 0.01)
#                        )[0]
#         ix_bout = ixs[0]
#         pose = angles_to_pose_names(angles[ix_bout], angle_names)
#         delay_ms = int(plot_delays[i]*1000)
#         sub = slice(300, 600) if dist == 'poisson' else slice(305, 400)
#         pdf = np.mean(pdfs_shaped[ix_bout, sub])
#         print('{}, {} ms actuation, {:.3f}'.format(dist, delay_ms, pdf))
#         make_fly_video(pose, '../vids/simulated_fly_actdelay_{}_{}ms.mp4'.format(dist, delay_ms))


# ### all the heatmap plots!
# 
# 

# In[ ]:


to_plot  = [#("during - before perturbations", "during_diff_logpdf"),
            # ("after - before perturbations", "after_diff_logpdf"),
            # ("before perturbations", "before_logpdf"),
            ("during perturbations", "during_logpdf"),
            ("after perturbations", "after_logpdf")
            ]


# In[ ]:


for dist_name, dist in [('continuous', DistType.POISSON_GAUSSIAN),
                        ('impulse', DistType.IMPULSE)]:

    if dist_name == 'continuous':
        before = np.mean(pdfs_shaped[:, :300], axis=1)
        during = np.mean(pdfs_shaped[:, 300:600], axis=1)
        after = np.mean(pdfs_shaped[:, 600:900], axis=1)
        # max_velocity_constant = 2.5
        max_velocity_constant = 1.875
    elif dist_name == 'impulse':
        before = np.mean(pdfs_shaped[:, :300], axis=1)
        during = np.mean(pdfs_shaped[:, 305:400], axis=1)
        after = np.mean(pdfs_shaped[:, 400:900], axis=1)
        # max_velocity_constant = 5.0
        max_velocity_constant = 5.0

    dd = pd.DataFrame({"act_delay": act_delays,
                   "dist_type": dist_types,
                   "sense_delay": sense_delays,
                   "speed": speeds[:, 0],
                   "max_velocity": max_velocities,
                   "during_logpdf": during,
                   "after_logpdf": after,
                   "during_diff_logpdf": during-before,
                   "after_diff_logpdf": after-before,
                   "before_logpdf": before})

    for xaxis in ['speed', 'max_velocity']:
        check = np.isclose(dd['sense_delay'], 0.01) & (dd['dist_type'] == dist.value)
        if xaxis == 'speed':
            check = check & np.isclose(dd['max_velocity'], max_velocity_constant)
        elif xaxis == 'max_velocity':
            check = check & np.isclose(dd['speed'], 12)
        dgroup = dd[check].groupby(['act_delay', xaxis]).mean()


        for (name, key) in to_plot:
            dimg = dgroup.reset_index().pivot(columns=xaxis, index='act_delay', values=key)
            plt.figure(figsize=(6, 3), dpi=200)
            plt.imshow(dimg)
            if "diff" in key:
                plt.imshow(dimg, vmin=-2, vmax=0, origin='lower')
            else:
                plt.imshow(dimg, vmin=-2, vmax=-1, origin='lower')

            ax = plt.gca()
            ax.set_xticks(np.arange(len(dimg.columns)), labels=dimg.columns)
            ax.set_yticks(np.arange(len(dimg.index)), labels=np.int32(dimg.index * 1000))

            # ax.invert_yaxis()

            if xaxis == 'speed':
                ax.set_xlabel("Speed (mm/s)")
            elif xaxis == 'max_velocity':
                ax.set_xlabel("Perturbation strength")
            ax.set_ylabel("Delay (ms)")

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            plt.colorbar()
            plt.title(name)

            plt.savefig('plots/actdelay_logpdf_{}_{}_{}.pdf'.format(dist_name, xaxis, key),
                        bbox_inches = "tight")


# ## Sensory delay plots
# 
# 

# sensory delay plots
# 
# -   [X] example time series at different delays
# -   [X] speed vs sensory delay plots
# -   [X] sensory delay vs perturbation strength plot
# 
# 

# ### Example time series
# 
# 

# For the figure part A, we'd like to have multiple example angles. Perhaps R1 femur-tibia flexion would be good to show, but also L2 femur rotation?
# I'd like to have multiple traces for each angle, perhaps we could show with forward, rotation, and sideslip?
# 
# In this code, we could also make a supplementary figure with a more complete set of angles.
# 
# 

# In[ ]:


# plot_speeds = [[[[12, 0, 0]]]]
plot_delays = [0, 0.005, 0.010, 0.015]
# plot_delays = [0, 0.004, 0.008, 0.012, 0.016]
# plot_velocities = [8, 14]
plot_speed = 12


# In[ ]:


angnames = ['R1C_flex', 'L2B_rot']
dists = ['poisson', 'impulse']
dist_values = {'poisson': DistType.POISSON_GAUSSIAN.value,
               'impulse': DistType.IMPULSE.value}

for dist in dists:
    dist_value = dist_values[dist]
    if dist == 'poisson':
        max_velocity_constant = 1.875
    else:
        max_velocity_constant = 5.0

    for angname in angnames:
        ix_ang = angle_names.index(angname)

        plt.figure(figsize=(5.5, 2))
        for i in range(len(plot_delays)):
            plt.subplot(len(plot_delays), 1, i+1)
            ixs = np.where((speeds[:, 0] == plot_speed)
                           & (dist_types == dist_value)
                           & np.isclose(sense_delays, plot_delays[i])
                           & np.isclose(max_velocities, max_velocity_constant)
                           & np.isclose(act_delays, 0.030)
                           )[0]
            ix_bout = ixs[1]
            print(conditions[ix_bout])
            ang = angles[ix_bout][:, ix_ang]
            t = np.arange(len(ang))/300.0
            if angname == 'R1C_flex':
                ang = np.clip(ang, 0, 180)
            elif angname == 'L2B_rot':
                ang = np.mod(ang, 360)
            plt.plot(t, ang)
            plt.axvline(t[300], color='gray', linestyle='dotted')
            plt.axvline(t[600], color='gray', linestyle='dotted')
            if angname == 'R1C_flex':
                plt.ylim(0, 180)
                plt.yticks([60, 120])
            else:
                plt.ylim(0, 360)
                plt.yticks([120, 240])
            if i != 2:
                plt.xticks(ticks=[])

        sns.despine(bottom=True)
        plt.ylabel("Angle (deg)")
        plt.xlabel("Time (s)")

        plt.savefig('plots/sense_delays_trace_{}_{}.pdf'.format(angname, dist),
                    bbox_inches = "tight")


# ### Videos
# 
# 

# In[ ]:


dists = ['poisson', 'impulse']
dist_values = {'poisson': DistType.POISSON_GAUSSIAN.value,
               'impulse': DistType.IMPULSE.value}

for dist in dists:
    dist_value = dist_values[dist]
    if dist == 'poisson':
        max_velocity_constant = 1.875
    else:
        max_velocity_constant = 5.0

    for i in range(len(plot_delays)):
        ixs = np.where((speeds[:, 0] == plot_speed)
                       & (dist_types == dist_value)
                       & np.isclose(sense_delays, plot_delays[i])
                       & np.isclose(max_velocities, max_velocity_constant)
                       & np.isclose(act_delays, 0.030)
                       )[0]
        ix_bout = ixs[0]
        pose = angles_to_pose_names(angles[ix_bout], angle_names)
        delay_ms = int(plot_delays[i]*1000)
        sub = slice(300, 600) if dist == 'poisson' else slice(305, 400)
        pdf = np.mean(pdfs_shaped[ix_bout, sub])
        print('{}, {} ms sensory, {:.3f}'.format(dist, delay_ms, pdf))
        # make_fly_video(pose, '../vids/simulated_fly_sensedelay_{}_{}ms.mp4'.format(dist, delay_ms))
        np.savez_compressed('../vids-npz/simulated_fly_sensedelay_{}_{}ms.npz'.format(dist, delay_ms), pose=pose)


# ### all the heatmap plots!
# 
# 

# In[ ]:


to_plot  = [# ("during - before perturbations", "during_diff_logpdf"),
            # ("after - before perturbations", "after_diff_logpdf"),
            # ("before perturbations", "before_logpdf"),
            ("during perturbations", "during_logpdf"),
            ("after perturbations", "after_logpdf")
            ]


# In[ ]:


for dist_name, dist in [('continuous', DistType.POISSON_GAUSSIAN),
                        ('impulse', DistType.IMPULSE)]:

    if dist_name == 'continuous':
        before = np.mean(pdfs_shaped[:, :300], axis=1)
        during = np.mean(pdfs_shaped[:, 300:600], axis=1)
        after = np.mean(pdfs_shaped[:, 600:900], axis=1)
        # max_velocity_constant = 2.5
        max_velocity_constant = 1.875
    elif dist_name == 'impulse':
        before = np.mean(pdfs_shaped[:, :300], axis=1)
        during = np.mean(pdfs_shaped[:, 305:400], axis=1)
        after = np.mean(pdfs_shaped[:, 400:900], axis=1)
        # max_velocity_constant = 5.0
        max_velocity_constant = 5.0

        
    dd = pd.DataFrame({"act_delay": act_delays,
                   "dist_type": dist_types,
                   "sense_delay": sense_delays,
                   "speed": speeds[:, 0],
                   "max_velocity": max_velocities,
                   "during_logpdf": during,
                   "after_logpdf": after,
                   "during_diff_logpdf": during-before,
                   "after_diff_logpdf": after-before,
                   "before_logpdf": before})

    for xaxis in ['speed', 'max_velocity']:
        check = np.isclose(dd['act_delay'], 0.030) & (dd['dist_type'] == dist.value)
        if xaxis == 'speed':
            check = check & np.isclose(dd['max_velocity'], max_velocity_constant)
        elif xaxis == 'max_velocity':
            check = check & np.isclose(dd['speed'], 12)
        dgroup = dd[check].groupby(['sense_delay', xaxis]).mean()


        for (name, key) in to_plot:
            dimg = dgroup.reset_index().pivot(columns=xaxis, index='sense_delay', values=key)
            plt.figure(figsize=(6, 3), dpi=200)
            if "diff" in key:
                plt.imshow(dimg, vmin=-2, vmax=0, origin='lower')
            else:
                plt.imshow(dimg, vmin=-2, vmax=-1, origin='lower')

            ax = plt.gca()
            ax.set_xticks(np.arange(len(dimg.columns)), labels=dimg.columns)
            ax.set_yticks(np.arange(len(dimg.index)), labels=np.int32(dimg.index * 1000))

            # ax.invert_yaxis()

            if xaxis == 'speed':
                ax.set_xlabel("Speed (mm/s)")
            elif xaxis == 'max_velocity':
                ax.set_xlabel("Perturbation strength")
            ax.set_ylabel("Delay (ms)")

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            plt.colorbar()
            plt.title(name)

            plt.savefig('plots/sensedelay_logpdf_{}_{}_{}.pdf'.format(dist_name, xaxis, key),
                        bbox_inches = "tight")


# In[ ]:





# In[ ]:





# In[ ]:




