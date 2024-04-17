#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

# fname = '/home/lili/data/tuthill/models/models_sls_sweep_v2/models_errors_speedhack_v2.csv'
fname = '/home/lili/data/tuthill/models/models_sls_sweep_v3/models_errors_speedhack_v2.csv'
# fname = '/home/lili/data/tuthill/models/models_sls_sweep_moreiters/models_errors_speedhack.csv'

data = pd.read_csv(fname)

data['speed_error'] = (data['speed_sim'] - data['speed_real']).abs()
data['step_length_error'] = (data['step_length_sim'] - data['step_length_real']).abs()
data['step_freq_error'] = (data['step_freq_sim'] - data['step_freq_real']).abs()

d = data.loc[data['dist'] & data['coupling_ratio'] == 1]
means = d.groupby(['model']).mean()
# sub_dist = means[['angle_error', 'deriv_error', 'ks_sim']]
sub_dist = means[['deriv_error']]

d = data.loc[~data['dist'] & data['coupling_ratio'] == 1]
means = d.groupby(['model']).mean()
# sub_nodist = means[['angle_error', 'deriv_error', 'ks_sim']]
sub_nodist = means[['deriv_error', 'step_freq_error', 'step_length_error', 'speed_error']]

# print(sub_nodist.to_markdown())
# print('\n\n')
# print(sub_dist.to_markdown())

sub_dist.columns = ['dist_deriv_error']
sub = pd.concat([sub_nodist, sub_dist], axis=1)
# print(sub.round(2).to_markdown())

# best = sub.sort_values(by='dist_deriv_error', ascending=True).head(n=20)
# best = sub.sort_values(by='dist_ks_sim', ascending=False).head(n=20)
# print(best.round(2).to_markdown())

# print()
# best = sub.sort_values(by='dist_ks_sim', ascending=False).head(n=20)
# # best = sub.sort_values(by='dist_ks_sim', ascending=False).head(n=20)
# print(best.round(2).to_markdown())

best = sub.sort_values(by='deriv_error', ascending=True).head(n=20)
# best = sub.sort_values(by='step_freq_error', ascending=True).head(n=20)
# # best = sub.sort_values(by='dist_ks_sim', ascending=False).head(n=20)


print(best.round(2).to_markdown())


# model = 'model_hd256_dr000_pn000_np000_ep1000.pkl'
# model = 'model_hd032_dr000_pn030_np000.pkl'
# model = 'model_hd128_dr010_pn000_np000_ep1000.pkl'
# model = 'model_hd016_dr010_pn110_np000_ep100.pkl'
# model = 'model_hd032_dr000_pn070_np000_ep100.pkl'
# model = 'model_hd064_dr000_pn010_np000_ep1000.pkl'
# model = 'model_hd256_dr005_pn020_np000_ep1000.pkl'
# model = 'model_hd128_dr010_pn010_np000_ep1000.pkl'
# model = 'model_hd032_dr000_pn030_np000.pkl'
#
# model = 'models_sls_sweep_v2/model_hd064_dr000_pn010_np000_ep100.pkl'
# model = 'models_sls_sweep_v2/model_hd128_dr000_pn000_np000_ep900.pkl'
# model = 'models_sls_sweep_v2/model_hd128_dr005_pn020_np000_ep900.pkl'
# model = 'models_sls_sweep_v2/model_hd064_dr005_pn020_np000_ep100.pkl'
# model = 'models_sls_sweep_v2/model_hd032_dr000_pn000_np000_ep1000.pkl'
#
# model = 'models_sls_sweep_v3/model_hd048_dr000_pn000_np000_ep100.pkl'
model = 'models_sls_sweep_v3/model_hd064_dr000_pn000_np000_ep900.pkl'
# model = 'models_sls_sweep_v3/model_hd048_dr005_pn010_np000_ep100.pkl'
# model = 'models_sls_sweep_v3/model_hd064_dr000_pn020_np000_ep100.pkl'
# model = 'models_sls_sweep_v3/model_hd032_dr000_pn040_np000_ep100.pkl'
d = data.loc[~data['dist'] & (data['coupling_ratio'] == 1) & (data['model'] == model)]


# import matplotlib
# matplotlib.use('TkAgg')


# plt.figure(1)
# plt.clf()
# plt.scatter(sub['deriv_error'], sub['dist_deriv_error'])
# plt.xlim(0, 400)
# plt.ylim(0, 400)
# plt.draw()
# plt.show(block=False)

x = 'speed'

plt.figure(1)
plt.clf()
# plt.scatter(d['speed_real'], d['speed_sim'])
plt.plot(d[x + '_sim'], d[x + '_sim'], color='black')
plt.scatter(d[x + '_real'], d[x + '_sim'])
# plt.scatter(d['speed_x'], d[x + '_real'])
plt.xlabel('speed real (mm/s)')
plt.ylabel('speed simulated (mm/s)')
plt.draw()
plt.show(block=False)
