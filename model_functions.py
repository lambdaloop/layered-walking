#!/usr/bin/env ipython

from scipy import stats, signal
import os
from scipy.special import expit, logit
from collections import Counter
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers

FPS = 300.0

def filter_bouts(bnums, inp):
  angles_raw, angles_deriv, bout_numbers, fictrac_vals, angle_names = inp
  ix = angle_names.index('L1C_flex')
  good_bouts = []
  for bnum in np.unique(bnums):
      if bnum == 0 or np.isnan(bnum): continue
      cc = np.isclose(bout_numbers, bnum)
      raw = angles_raw[cc, ix]
      deriv = angles_deriv[cc, ix] / FPS
      low, high = np.percentile(raw, [5, 95])
      high_deriv = np.percentile(deriv, 95)
      vals = fictrac_vals[cc]
      check = np.all(np.isfinite(vals))
      if check and np.mean(vals[:,0] > 1.2) > 0.5 and \
         high - low > 40 and high_deriv > 4 and len(raw) >= 150:
          good_bouts.append(bnum)
  good_bouts = np.array(good_bouts)
  return good_bouts

def get_period(ang):
    peaks, _ = signal.find_peaks(ang, height=80, distance=10)
    period = np.zeros(len(ang))
    for p1, p2 in zip(peaks, peaks[1:]):
        period[p1:p2] = p2 - p1
    period[period == 0] = np.median(period[period > 0])
    return period

def get_speed(ang):
    return 1.0 / get_period(ang)


gcamp_fps = 8.01
t = np.arange(0, 3.0, 1.0/gcamp_fps)
tau_on = 0.07
tau_off = 0.38
gcamp = np.exp(-t / tau_off) - np.exp(-t / tau_on)
gcamp = gcamp / np.sum(gcamp)


def predict_signal_basic(params, ang):
    a3, a2, a1, a0 = params[:4]
    tau_sig = np.exp(params[4]) + 1e-5
    # make the filters
    ff = np.exp(-t / tau_sig)
    ff = ff / np.sum(ff)
    # apply the filters
    n_pad = len(t) + 10
    ang_pad = np.pad(ang, n_pad, mode='edge')
    # linear = signal.convolve(ang_pad, ff, mode='full')
    linear = ang_pad
    nonlinear = np.power(linear, 3)*a3 + \
        np.power(linear, 2)*a2 + linear*a1 + a0
    # pred = signal.convolve(nonlinear, gcamp, mode='full')
    pred = nonlinear
    pred = pred[n_pad:len(ang)+n_pad]
    return pred


def predict_signal_squash(params, ang):
    a3, a2, a1, a0 = params[:4]
    mean = params[4]
    threshold = params[5]
    tau_sig = np.exp(params[6]) + 1e-5
    # make the filters
    ff = np.exp(-t / tau_sig)
    ff = ff / np.sum(ff)
    # apply the filters
    n_pad = len(t) + 10
    ang_pad = np.pad(ang, n_pad, mode='edge')
    # linear = signal.convolve(ang_pad, ff, mode='full')
    linear = ang_pad
    nonlinear = np.power(linear, 3)*a3 + \
        np.power(linear, 2)*a2 + linear*a1 + a0
    nonlinear = threshold * expit((nonlinear-mean)/threshold) + mean
    # pred = signal.convolve(nonlinear, gcamp, mode='full')
    pred = nonlinear
    pred = pred[n_pad:len(ang)+n_pad]
    return pred


def run_bouc_wen(x, params):
    z = x[0]
    zs = np.zeros(len(x))
    zs[0] = z
    rho_1, rho_2 = expit(params[:2])
    alpha, beta, eta = params[2:5]
    for i in range(1, len(x)):
        dx = x[i] - rho_2 * x[i-1]
        z = rho_1*z + alpha*dx - beta*np.abs(dx)*z - eta*dx*np.abs(z)
        zs[i] = z
    return zs

def predict_signal_hysteresis(params, ang):
    a3, a2, a1, a0 = params[:4]
    mean, std = params[4:6]
    bouc_wen_params = params[6:]
    # apply the filters
    n_pad = len(t) + 10
    ang_pad = np.pad(ang, n_pad, mode='edge')
    # linear = signal.convolve(ang_pad, x, mode='full')
    ang_pad_norm = (ang_pad - mean) / std
    linear = run_bouc_wen(ang_pad_norm, bouc_wen_params)
    linear[~np.isfinite(linear)] = 0
    linear[np.abs(linear) > 1e5] = 0
    nonlinear = np.power(linear, 3)*a3 + \
        np.power(linear, 2)*a2 + linear*a1 + a0
    # pred = signal.convolve(nonlinear, gcamp, mode='full')
    pred = nonlinear
    pred = pred[n_pad:len(ang)+n_pad]
    return pred



claw_flex_hysteresis = np.array([-2.931227436119279e-06, 0.0008805231536923304,
                                 -0.07500382910326384, 1.2181527080908428, 0.6998556675593381, 1.007719843402409,
                                 4.296173813194425, 5.869956564066783, 0.9498351635865095, -0.0079321688577881,
                                 -0.0011331643323440244])
claw_ext_hysteresis = np.array([-5.624181112108334e-07, 0.0002676035983711878,
                                -0.020917851912373952, -0.34458646329587317, 0.29811438534731055,
                                1.0086571425019881, 3.339400868061741, 4.092325773442487, 0.817091971639576,
                                0.009392102573046748, -0.01987544630376564])

hook_flex_squash = np.array([-0.7419521043329518, 1.6852463336631625,
                             11.038491144496458, -54.85481492993002, -0.40723310570725896, 10.58916549474057,
                             -2.995732273553991])

club_squash = np.array([-0.04212216385518028, 1.822561122997191,
                        0.48463394824183065, -44.49827166271501, -0.030528575786051768,
                        6.6386739643618045, -2.995732273553991])


claw_ext_basic = np.array([8.223901568251783e-07, 6.34182987527203e-05,
                           -0.02313163565546027, 0.25882194947607556, -2.995732273553991])

claw_flex_basic = np.array([-1.952668012508767e-06, 0.0009096231582972422,
                            -0.1274669444523227, 4.896347729401412, -2.995732273553991])


prop_params = (claw_flex_basic, claw_ext_basic, club_squash, hook_flex_squash)

def get_props(ang):
    deriv = signal.savgol_filter(ang, 5, 2, deriv=1)

    pp = np.zeros((len(ang), 5))
    # pp[:,0] = predict_signal_hysteresis(claw_flex_hysteresis, ang)
    # pp[:,1] = predict_signal_hysteresis(claw_ext_hysteresis, ang)
    pp[:,0] = predict_signal_basic(claw_flex_basic, ang)
    pp[:,1] = predict_signal_basic(claw_ext_basic, ang)
    pp[:,2] = predict_signal_squash(club_squash, deriv)
    pp[:,3] = predict_signal_squash(hook_flex_squash, deriv)
    pp[:,4] = predict_signal_squash(hook_flex_squash, -1 * deriv) # hook ext
    return pp


def get_props_by_bouts(ang, bnums):
    out = np.zeros((len(ang), 5))
    for bnum in np.unique(bnums):
        # c = np.isclose(bnums, bnum)
        c = bnums == bnum
        out[c] = get_props(ang[c])
    return out



def wrap_array(x):
    if len(x.shape) == 1:
        return x[:, None]
    else:
        return x

def summarize(x):
    return np.mean(x, axis=0), np.std(x, axis=0)

def get_walk_xy(inp, props, phases, context, fnum, fname, use_phase=True, direct_props=False):
    raw = wrap_array(inp[:,0])
    deriv = wrap_array(inp[:,1])
    accel = wrap_array(inp[:,2])
    phase_raw = wrap_array(phases[:,0])
    phase_deriv = wrap_array(phases[:,1])
    context = wrap_array(context)
    check = (fname[1:] == fname[:-1]) & (fnum[1:]-1 == fnum[:-1])
    derivf = deriv / FPS
    accelf = accel / (FPS * FPS)

    if direct_props:
        xx = props
    else:
        xx = np.hstack([raw, derivf])

    if use_phase:
        x_walk = np.hstack([xx, context,
                            np.cos(phase_raw), np.sin(phase_raw)])[1:]
        y_walk = np.hstack([accelf, phase_deriv])[1:]
    else:
        x_walk = np.hstack([xx, context])[1:]
        y_walk = np.hstack([accelf])[1:]

    x_walk = x_walk[check].astype('float32')
    y_walk = y_walk[check].astype('float32')
    msx_w = summarize(x_walk)
    msy_w = summarize(y_walk)

    return (x_walk, y_walk, msx_w, msy_w)

def get_state_xy(inp, props, context, fnum, fname, use_state=True):
    raw = wrap_array(inp[:,0])
    deriv = wrap_array(inp[:,1])
    accel = wrap_array(inp[:,2])
    context = wrap_array(context)
    check = (fname[1:] == fname[:-1]) & (fnum[1:]-1 == fnum[:-1])
    derivf = deriv / FPS
    accelf = accel / (FPS * FPS)

    if use_state:
        x_state = np.hstack([raw, derivf, context])[:-1]
    else:
        x_state = np.hstack([context])[:-1]
    x_state = np.hstack([props[1:], x_state])
    y_state = np.hstack([raw, derivf])[1:]

    x_state = x_state[check].astype('float32')
    y_state = y_state[check].astype('float32')
    msx_s = summarize(x_state)
    msy_s = summarize(y_state)

    return (x_state, y_state, msx_s, msy_s)

def get_sw_xy(inp, props, phases, context, fnum, fname, params):
    out_walk = get_walk_xy(inp, props, phases, context, fnum, fname,
                           use_phase=params.get('use_phase', True),
                           direct_props=params.get('direct_props', False))
    out_state = get_state_xy(inp, props, context, fnum, fname,
                             use_state=params.get('use_state', True))
    return out_state, out_walk




class MLPScaledXY(Model):
  def __init__(self, output_dim=10, hidden_dim=512, dropout_rate=None,
               msx=(0, 1), msy=(0, 1)):
    super().__init__()
    self.hidden1 = layers.Dense(hidden_dim, name="hidden1")
    self.hidden2 = layers.Dense(hidden_dim, name="hidden2")
    self.final = layers.Dense(output_dim, name="final")
    self.msx = msx
    self.msy = msy
    self._dropout_rate = dropout_rate
    self._output_dim = output_dim
    self._hidden_dim = hidden_dim

  def call(self, x, is_training=False):
    use_dropout = self._dropout_rate not in (None, 0)
    mx, sx = self.msx
    my, sy = self.msy
    xs = (x - mx) / sx
    output = tf.nn.elu(self.hidden1(xs))
    if is_training and use_dropout:
      output = tf.nn.dropout(output, self._dropout_rate)
    output = tf.nn.elu(self.hidden2(output))
    if is_training and use_dropout:
      output = tf.nn.dropout(output, self._dropout_rate)
    output = self.final(output)
    output = output * sy + my
    return output

  def get_config(self):
    return {
      'dropout_rate': self._dropout_rate,
      'output_dim': self._output_dim,
      'hidden_dim': self._hidden_dim,
      'msx': self.msx,
      'msy': self.msy
    }

  def from_config(config):
    return MLPScaledXY(**config)

  def get_full(self):
    return {'config': self.get_config(),
            'weights': self.get_weights()}

  def from_full(full):
    new = MLPScaledXY.from_config(full['config'])
    new.build(full['weights'][0].T.shape) # initialize dimension
    new.set_weights(full['weights'])
    return new


def num_vars(model):
    return sum([x.numpy().size for x in model.trainable_variables])
