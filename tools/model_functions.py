#!/usr/bin/env ipython

from scipy import stats, signal
import os
from scipy.special import expit, logit
from collections import Counter
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress warnings from tensorflow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # compute on cpu, it's actually faster for inference with smaller model

import numpy as np
import xarray as xr
from typing import Optional, Text

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

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

def predict_signal_basic_quartic(params, ang):
    a4, a3, a2, a1, a0 = params[:5]
    n_pad = len(t) + 10
    ang_pad = np.pad(ang, n_pad, mode='edge')
    linear = ang_pad
    nonlinear = np.power(linear, 4)*a4 + np.power(linear, 3)*a3 + \
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


hook_flex_new = np.array([-1.55663187e-05,  2.09135548e-03,  3.46524646e-03, -8.96695952e-01, -1.66111787e+01])
hook_ext_new = np.array([-6.08388002e-04, -3.50381872e-03,  1.72046329e-01,  6.85051337e-01, -4.92352479e+00])


prop_params = (claw_flex_basic, claw_ext_basic, club_squash, hook_flex_squash)
prop_params_new = (claw_flex_basic, claw_ext_basic, club_squash, hook_flex_new, hook_ext_new)

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

def get_props_new(ang):
    deriv = signal.savgol_filter(ang, 5, 2, deriv=1)

    pp = np.zeros((len(ang), 5))
    # pp[:,0] = predict_signal_hysteresis(claw_flex_hysteresis, ang)
    # pp[:,1] = predict_signal_hysteresis(claw_ext_hysteresis, ang)
    pp[:,0] = predict_signal_basic(claw_flex_basic, ang)
    pp[:,1] = predict_signal_basic(claw_ext_basic, ang)
    pp[:,2] = predict_signal_squash(club_squash, deriv)
    pp[:,3] = predict_signal_basic_quartic(hook_flex_new, deriv)
    pp[:,4] = predict_signal_basic_quartic(hook_ext_new, deriv)
    return pp


def get_props_by_bouts(ang, bnums, model='old'):
    out = np.zeros((len(ang), 5))
    if model == 'old':
      fun = get_props
    elif model == 'new':
      fun = get_props_new
    else:
      raise ValueError("Invalid model name: {}".format(model))
    for bnum in np.unique(bnums):
        # c = np.isclose(bnums, bnum)
        c = bnums == bnum
        out[c] = fun(ang[c])
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



class ModeAdaptiveLinear(Model):
  """Linear module, modified to handle phase as an input."""

  def __init__(self,
               output_size: int,
               n_control: int = 4,
               with_bias: bool = True,
               name: Optional[Text] = None):
    super(ModeAdaptiveLinear, self).__init__(name=name)
    self.output_size = output_size
    self.with_bias = with_bias
    self.n_control = n_control

  def build(self, input_shape):
      self.w = self.add_weight(shape=(self.n_control, input_shape[-1], self.output_size),
                               initializer='random_normal',
                               trainable=True)
      if self.with_bias:
        self.b = self.add_weight(shape=(self.n_control, self.output_size,),
                                 initializer='zeros',
                                 trainable=True)

  @tf.function
  def _get_weights(self, weights: tf.Tensor):
    w_p = tf.einsum('ji,imk->jmk', weights, self.w)
    if self.with_bias:
      b_p = tf.einsum('ji,ik->jk', weights, self.b)
    else:
      b_p = None
    return w_p, b_p

  @tf.function
  def call(self, inputs: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    w_p, b_p = self._get_weights(weights)

    outputs = tf.squeeze(tf.matmul(tf.expand_dims(inputs, 1), w_p))
    if self.with_bias:
      outputs = tf.add(outputs, b_p)
    return outputs


class PFMLPScaledXY(Model):
  def __init__(self, output_dim=10, hidden_dim=512, dropout_rate=None,
               msx=(0, 1), msy=(0, 1)):
    super().__init__()
    self.hidden1 = ModeAdaptiveLinear(hidden_dim, n_control=2, name="hidden1")
    self.hidden2 = ModeAdaptiveLinear(hidden_dim, n_control=2, name="hidden2")
    self.final = ModeAdaptiveLinear(output_dim, n_control=2, name="final")
    self.msx = msx
    self.msy = msy
    self._dropout_rate = dropout_rate
    self._output_dim = output_dim
    self._hidden_dim = hidden_dim

  @tf.function
  def call(self, x, is_training=False):
    use_dropout = self._dropout_rate not in (None, 0)
    phase_weights = x[..., -2:]
    mx, sx = self.msx
    my, sy = self.msy
    xs = (x - mx) / sx
    xs_other = xs[..., :-2]
    output = tf.nn.elu(self.hidden1(xs_other, phase_weights))
    if is_training and use_dropout:
      output = tf.nn.dropout(output, self._dropout_rate)
    output = tf.nn.elu(self.hidden2(output, phase_weights))
    if is_training and use_dropout:
      output = tf.nn.dropout(output, self._dropout_rate)
    output = self.final(output, phase_weights)
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
    return PFMLPScaledXY(**config)

  def get_full(self):
    return {'config': self.get_config(),
            'weights': self.get_weights()}

  def from_full(full):
    new = PFMLPScaledXY.from_config(full['config'])
    new.build(full['weights'][0].T.shape) # initialize dimension
    new.set_weights(full['weights'])
    return new

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

  @tf.function
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


def map_angle_basic(params, ang):
    a3, a2, a1, a0 = params[:4]
    linear = ang
    nonlinear = np.power(linear, 3)*a3 + \
        np.power(linear, 2)*a2 + linear*a1 + a0
    return nonlinear

def map_angle_quartic(params, ang):
    a4, a3, a2, a1, a0 = params[:5]
    linear = ang
    nonlinear = np.power(linear, 4)*a4 + np.power(linear, 3)*a3 + \
        np.power(linear, 2)*a2 + linear*a1 + a0
    return nonlinear


def map_angle_squash(params, ang):
    a3, a2, a1, a0 = params[:4]
    mean = params[4]
    threshold = params[5]
    linear = ang
    nonlinear = np.power(linear, 3)*a3 + \
        np.power(linear, 2)*a2 + linear*a1 + a0
    nonlinear = threshold * expit((nonlinear-mean)/threshold) + mean
    return nonlinear

def run_model(n_pred, init, context, prop_params, models, model_params, perturbations=[]):
    # init: angle, deriv, phase cos, phase sin
    claw_flex_basic, claw_ext_basic, club_squash, hook_flex_squash = prop_params
    model_state, model_walk = models

    # joint angles of the fly in this simulation
    angpred = np.zeros((n_pred, 2), dtype='float32')
    angpred[0] = init[:2]
    currang = np.copy(angpred[0])

    # proprioceptor responses
    ps = np.zeros((n_pred,5), dtype='float32') # actual proprioceptor responses
    a, d = angpred[0]
    ps[0,0] = map_angle_basic(claw_flex_basic, a)
    ps[0,1] = map_angle_basic(claw_ext_basic, a)
    ps[0,2] = map_angle_squash(club_squash, d)
    ps[0,3] = map_angle_squash(hook_flex_squash, d)
    ps[0,4] = map_angle_squash(hook_flex_squash, -1*d)

    # fly state estimator angles
    anghat = np.zeros((n_pred, 2), dtype='float32')
    anghat[0] = init[:2]

    # walking model phase
    phases_pred = np.zeros(n_pred, dtype='float32')
    if model_params['use_phase']:
      c, s = init[2:4]
      phases_pred[0] = np.mod(np.arctan2(s, c), 2*np.pi)

    # outputs from walking model
    accels = np.zeros(n_pred, dtype='float32')
    pderivs = np.zeros(n_pred, dtype='float32')

    for i in range(n_pred-1):
        if model_params.get('direct_props', False):
            pred_state = ps[i]
        else:
            if model_params.get('use_state', True):
                inp_state = np.hstack([ps[i], anghat[i], context[i]])
            else:
                inp_state = np.hstack([ps[i], context[i]])
            pred_state = model_state(inp_state[None])[0].numpy()
            pred_state[0] = np.clip(pred_state[0], 0, 180)
            anghat[i+1] = pred_state

        theta = phases_pred[i]
        if model_params['use_phase']:
            inp_walk = np.hstack([pred_state, context[i+1],
                                  np.cos(theta), np.sin(theta)])
            accel, pderiv = model_walk(inp_walk[None])[0]
        else:
            inp_walk = np.hstack([pred_state, context[i+1]])
            accel = model_walk(inp_walk[None])[0]
            pderiv = 0

        # accel, pderiv = out_walk[i]

        accels[i] = accel
        pderivs[i] = pderiv

        phases_pred[i+1] = np.mod(phases_pred[i] + pderiv, 2*np.pi)

        currang[0] = np.clip(currang[0] + angpred[i, 1] , 0, 180) # update angle
        currang[1] = np.clip(currang[1] + accel, -10, 10) # update deriv

        # currang = real_ang[i+1]

        angpred[i+1] = currang

        a, d = currang
        currp = np.zeros(5)
        currp[0] = map_angle_basic(claw_flex_basic, a)
        currp[1] = map_angle_basic(claw_ext_basic, a)
        currp[2] = map_angle_squash(club_squash, d)
        currp[3] = map_angle_squash(hook_flex_squash, d)
        currp[4] = map_angle_squash(hook_flex_squash, -1*d)

        for px in perturbations:
            if i >= px['start'] and i < px['end']:
                if 'value' in px:
                    currp[px['ix']] = px['value']
                elif 'add' in px:
                    currp[px['ix']] = currp[px['ix']] + px['add']

        ps[i+1] = currp

    out = {
        'angle': angpred,
        'phases': phases_pred,
        'anghat': anghat,
        'accel': accels,
        'phase_deriv': pderivs,
        'props': ps,
        'context': context
    }
    return out

def run_model_midpoint(n_pred, init, context, prop_params, models, model_params, perturbations=[]):
    # init: angle, deriv, phase cos, phase sin
    claw_flex_basic, claw_ext_basic, club_squash, hook_flex_new, hook_ext_new = prop_params
    model_state, model_walk = models

    # joint angles of the fly in this simulation
    angpred = np.zeros((n_pred, 2), dtype='float32')
    angpred[0] = init[:2]
    currang = np.copy(angpred[0])

    # proprioceptor responses
    ps = np.zeros((n_pred,5), dtype='float32') # actual proprioceptor responses
    a, d = angpred[0]
    ps[0,0] = map_angle_basic(claw_flex_basic, a)
    ps[0,1] = map_angle_basic(claw_ext_basic, a)
    ps[0,2] = map_angle_squash(club_squash, d*300.0 / 8.01)
    ps[0,3] = map_angle_quartic(hook_flex_new, d*300.0 / 200.0)
    ps[0,4] = map_angle_quartic(hook_ext_new, -1*d*300.0 / 200.0)

    # fly state estimator angles
    anghat = np.zeros((n_pred, 2), dtype='float32')
    anghat[0] = init[:2]

    # walking model phase
    phases_pred = np.zeros(n_pred, dtype='float32')
    if model_params['use_phase']:
      c, s = init[2:4]
      phases_pred[0] = np.mod(np.arctan2(s, c), 2*np.pi)

    # outputs from walking model
    accels = np.zeros(n_pred, dtype='float32')
    pderivs = np.zeros(n_pred, dtype='float32')

    for i in range(n_pred-1):
        if model_params.get('direct_props', False):
            pred_state = ps[i]
        else:
            if model_params.get('use_state', True):
                inp_state = np.hstack([ps[i], anghat[i], context[i]])
            else:
                inp_state = np.hstack([ps[i], context[i]])
            pred_state = model_state(inp_state[None])[0].numpy()
            pred_state[0] = np.clip(pred_state[0], 0, 180)
            anghat[i+1] = pred_state

        theta = phases_pred[i]
        if model_params['use_phase']:
            inp_walk = np.hstack([pred_state, context[i+1],
                                  np.cos(theta), np.sin(theta)])
            accel1, pderiv1 = model_walk(inp_walk[None])[0]
            ang1 = pred_state[0] + pred_state[1]*0.5
            drv1 = pred_state[1] + accel1*0.5
            phase1 = theta + pderiv1*0.5
            new_inp = np.hstack([ang1, drv1, context[i+1], np.cos(phase1), np.sin(phase1)])
            accel, pderiv = model_walk(new_inp[None].astype('float32'))[0].numpy()
        else:
            inp_walk = np.hstack([pred_state, context[i+1]])
            accel1 = model_walk(inp_walk[None])[0]
            ang1 = pred_state[0] + pred_state[1]*0.5
            drv1 = pred_state[1] + accel1*0.5
            new_inp = np.hstack([ang1, drv1, context[i+1]])
            accel = model_walk(new_inp[None].astype('float32'))[0].numpy()
            pderiv = 0

        # accel, pderiv = out_walk[i]

        accels[i] = accel
        pderivs[i] = pderiv

        phases_pred[i+1] = np.mod(phases_pred[i] + pderiv, 2*np.pi)

        # currang[0] = np.clip(currang[0] + angpred[i, 1] , 0, 180) # update angle
        # currang[1] = np.clip(currang[1] + accel, -10, 10) # update deriv

        currang[0] = np.clip(currang[0] + currang[1] , 0, 180) # update angle
        currang[1] = currang[1] + accel # update deriv

        # currang = real_ang[i+1]

        angpred[i+1] = currang

        a, d = currang
        currp = np.zeros(5)
        currp[0] = map_angle_basic(claw_flex_basic, a)
        currp[1] = map_angle_basic(claw_ext_basic, a)
        currp[2] = map_angle_squash(club_squash, d*300.0 / 8.01)
        currp[3] = map_angle_quartic(hook_flex_new, d*300.0 / 200.0)
        currp[4] = map_angle_quartic(hook_ext_new, -1*d*300.0 / 200.0)

        for px in perturbations:
            if i >= px['start'] and i < px['end']:
                if 'value' in px:
                    currp[px['ix']] = px['value']
                elif 'add' in px:
                    currp[px['ix']] = currp[px['ix']] + px['add']

        ps[i+1] = currp

    out = {
        'angle': angpred,
        'phases': phases_pred,
        'anghat': anghat,
        'accel': accels,
        'phase_deriv': pderivs,
        'props': ps,
        'context': context
    }
    return out


def run_model_midpoint_batch(n_pred, init, context, model, perturbations=[]):
    # init: angle, deriv, phase cos, phase sin
    claw_flex_basic, claw_ext_basic, club_squash, hook_flex_new, hook_ext_new = model['prop_params']
    model_state = model['model_state_loaded']
    model_walk = model['model_walk_loaded']
    model_params = model['params']

    pix = subset_props(model_params.get('prop_type', 'claw_hook_club'))

    n_batch = init.shape[0]

    # joint angles of the fly in this simulation
    angpred = np.zeros((n_pred, n_batch, 2), dtype='float32')
    angpred[0] = init[:,:2]
    currang = np.copy(angpred[0])

    # proprioceptor responses
    ps = np.zeros((n_pred,n_batch,5), dtype='float32') # actual proprioceptor responses
    a, d = angpred[0].T
    ps[0,:,0] = map_angle_basic(claw_flex_basic, a)
    ps[0,:,1] = map_angle_basic(claw_ext_basic, a)
    ps[0,:,2] = map_angle_squash(club_squash, d*300.0 / 8.01)
    ps[0,:,3] = map_angle_quartic(hook_flex_new, d*300.0 / 200.0)
    ps[0,:,4] = map_angle_quartic(hook_ext_new, -1*d*300.0 / 200.0)

    # fly state estimator angles
    anghat = np.zeros((n_pred, n_batch, 2), dtype='float32')
    anghat[0] = init[:,:2]

    # walking model phase
    phases_pred = np.zeros((n_pred, n_batch), dtype='float32')
    if model_params['use_phase']:
      c, s = init[:, -2:].T
      phases_pred[0] = np.mod(np.arctan2(s, c), 2*np.pi)

    # outputs from walking model
    accels = np.zeros((n_pred, n_batch), dtype='float32')
    pderivs = np.zeros((n_pred, n_batch), dtype='float32')

    for i in range(n_pred-1):
        if model_params.get('direct_props', False):
            pred_state = ps[i][:, pix]
        else:
            if model_params.get('use_state', True):
                inp_state = np.hstack([ps[i][:, pix], anghat[i], context[i]])
            else:
                inp_state = np.hstack([ps[i][:, pix], context[i]])
            pred_state = model_state(inp_state).numpy()
            pred_state[:,0] = np.clip(pred_state[:,0], 0, 180)
            anghat[i+1] = pred_state


        if model_params['use_phase']:
            theta = phases_pred[i][:,None]
            inp_walk = np.hstack([pred_state, context[i+1],
                                  np.cos(theta), np.sin(theta)])
            out = model_walk(inp_walk)
            accel1, pderiv1 = out[:,0:1], out[:,1:2]
            ang1 = pred_state[:,0:1] + pred_state[:,1:2]*0.5
            drv1 = pred_state[:,1:2] + accel1*0.5
            phase1 = theta + pderiv1*0.5
            new_inp = np.hstack([ang1, drv1, context[i+1], np.cos(phase1), np.sin(phase1)])
            out = model_walk(new_inp).numpy()
            accel, pderiv = out[:,0], out[:,1]
        else:
            inp_walk = np.hstack([pred_state, context[i+1]])
            accel1 = model_walk(inp_walk)
            ang1 = pred_state[:,0:1] + pred_state[:,1:2]*0.5
            drv1 = pred_state[:,1:2] + accel1*0.5
            new_inp = np.hstack([ang1, drv1, context[i+1]])
            accel = model_walk(new_inp).numpy().ravel()
            pderiv = 0

        # accel, pderiv = out_walk[i]

        accels[i] = accel
        pderivs[i] = pderiv

        phases_pred[i+1] = np.mod(phases_pred[i] + pderiv, 2*np.pi)

        # currang[0] = np.clip(currang[0] + angpred[i, 1] , 0, 180) # update angle
        # currang[1] = np.clip(currang[1] + accel, -10, 10) # update deriv

        currang[:,0] = np.clip(currang[:,0] + currang[:,1] , 0, 180) # update angle
        currang[:,1] = currang[:,1] + accel # update deriv

        # currang = real_ang[i+1]

        angpred[i+1] = currang

        a, d = currang.T
        currp = np.zeros((n_batch, 5))
        currp[:,0] = map_angle_basic(claw_flex_basic, a)
        currp[:,1] = map_angle_basic(claw_ext_basic, a)
        currp[:,2] = map_angle_squash(club_squash, d*300.0 / 8.01)
        currp[:,3] = map_angle_quartic(hook_flex_new, d*300.0 / 200.0)
        currp[:,4] = map_angle_quartic(hook_ext_new, -1*d*300.0 / 200.0)

        for bnum in range(n_batch):
          for px in perturbations[bnum]:
              if i >= px['start'] and i < px['end']:
                  if 'value' in px:
                      currp[bnum, px['ix']] = px['value']
                  elif 'add' in px:
                      currp[bnum, px['ix']] = currp[bnum, px['ix']] + px['add']

        ps[i+1] = currp

    out = {
        'angle': angpred,
        'phases': phases_pred,
        'anghat': anghat,
        'accel': accels,
        'phase_deriv': pderivs,
        'props': ps,
        'context': context
    }
    return out


def output_to_xarray(out, params):
    d = dict()
    for k in ['angle', 'anghat']:
        d[k] = xr.DataArray(out[k], dims=['time', 'run', 'ang_deriv'],
                            coords={'ang_deriv': ['angle', 'deriv']})
    for k in ['phases', 'accel', 'phase_deriv']:
        d[k] = xr.DataArray(out[k], dims=['time', 'run'])
    d['props'] = xr.DataArray(out['props'], dims=['time', 'run', 'prop_name'],
                              coords={'prop_name': ['claw_flex', 'claw_ext',
                                                    'club', 'hook_flex', 'hook_ext']})
    d['context'] = xr.DataArray(out['context'], dims=['time', 'run', 'context_name'],
                                coords={'context_name': params['context']})
    return xr.Dataset(d)

def load_models(m):
    if 'model_walk_loaded' not in m:
        m['model_walk_loaded'] = MLPScaledXY.from_full(m['model_walk'])
    if 'model_state' in m and 'model_state_loaded' not in m:
        m['model_state_loaded'] = MLPScaledXY.from_full(m['model_state'])
    else:
        m['model_state_loaded'] = None
    return m


def get_model_input(m, bout_number, offset, n_pred, fake_context=None, data_type='test'):
    (xy_s_test, xy_w_test), bnums_test = m[data_type]

    m = load_models(m)

    model_walk = m['model_walk_loaded']
    model_state = m['model_state_loaded']

    models = model_state, model_walk

    prop_params = m['prop_params']
    params = m['params']

    # cc = np.isclose(bnums_test, bout_number)
    cc = bnums_test == bout_number

    in_state = xy_s_test[0][cc][offset:]
    out_state = xy_s_test[1][cc][offset:]
    in_walk = xy_w_test[0][cc][offset:]
    out_walk = xy_w_test[1][cc][offset:]
    extra_walk = in_walk[:, 2:]

    # the commands sent down, this will be c(t-1) here
    # context_ref = in_state[:,-1:]
    context_ref = extra_walk[:,2:3]
    if fake_context is not None:
        context = np.tile([fake_context], (n_pred,1)).astype('float32')
    else:
        context = context_ref
        n_pred = len(context)

    # init: angle, deriv, phase cos, phase sin
    if params['use_phase']:
      init = np.hstack([out_state[0], extra_walk[0,-2:]])
    else:
      init = np.copy(out_state[0])

    stuff = n_pred, init, context, prop_params, models, params
    return stuff


def subset_data_props(xy_s, prop_type):
  if prop_type == "claw":
    xy_s = list(xy_s)
    xy_s[0] = np.hstack([xy_s[0][:, :2], xy_s[0][:, 5:]])
    xy_s[2] = (np.hstack([xy_s[2][0][:2], xy_s[2][0][5:]]),
               np.hstack([xy_s[2][1][:2], xy_s[2][1][5:]]))
  elif prop_type == "claw_hook":
    xy_s = list(xy_s)
    xy_s[0] = np.hstack([xy_s[0][:, :2], xy_s[0][:, 3:]]) # no club
    xy_s[2] = (np.hstack([xy_s[2][0][:2], xy_s[2][0][3:]]),
               np.hstack([xy_s[2][1][:2], xy_s[2][1][3:]]))
  elif prop_type == "claw_hook_club":
    pass
  else:
    raise ValueError("""Invalid value for variable prop_type: {}.
  Should be one of ['claw', 'claw_hook', 'claw_hook_club']""".format(prop_type))
  return xy_s

def subset_props(prop_type):
  if prop_type == "claw":
    return [0, 1]
  elif prop_type == "claw_hook":
    return [0, 1, 3, 4]
  elif prop_type == "claw_hook_club":
    return [0, 1, 2, 3, 4]
  else:
    raise ValueError("""Invalid value for variable prop_type: {}.
  Should be one of ['claw', 'claw_hook', 'claw_hook_club']""".format(prop_type))
