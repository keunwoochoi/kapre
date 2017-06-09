# -*- coding: utf-8 -*-
"""
Utils
=====
"""
from __future__ import absolute_import
import numpy as np
from keras.engine import Layer
from keras import backend as K
from . import backend
from . import backend_keras


class AmplitudeToDB(Layer):
    '''A layer that converts amplitude to decibel

    Parameters
    ----------
    amin: float [scalar]
        Noise floor.
        
    top_db: float [scalar]
        Dynamic range of output.

    Example
    -------
    Adding ``AmplitudeToDB`` after a spectrogram::

        model.add(Spectrogram(return_decibel=False))
        model.add(AmplitudeToDB())


    '''

    def __init__(self, ref_power=1.0, amin=1e-10, top_db=80.0):
        assert isinstance(ref_power, float) or ref_power == 'max'
        self.ref_power = ref_power
        self.amin = amin
        self.top_db = top_db
        super(AmplitudeToDB, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return backend_keras.amplitude_to_decibel(x, self.ref_power, self.amin, self.top_db)

    def get_config(self):
        config = {'ref_power': self.ref_power,
                  'amin': self.amin,
                  'top_db': self.top_db}
        base_config = super(AmplitudeToDB, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Normalization2D(Layer):
    '''A layer that normalises input data in ``axis`` axis.

    Parameters
    ----------
    input_shape: tuple
        E.g., ``(None, n_ch, n_row, n_col)`` if theano.

    int_axis: int
        |  axis index that along which mean/std is computed.
        |  0 for per data sample, -1 for per batch.
        |  1, 2, 3 for channel, row, col (if channels_first)
        |  if ``None``, ``str_axis`` SHOULD BE set.

    str_axis: str
        |  used ONLY IF ``int_axis`` is ``None``.
        |  ``'batch'``, ``'data_sample'``, ``'channel'``, ``'freq'``, ``'time')``
        |  Even though it is optional, actually it is recommended to use
        |  ``str_axis`` over ``int_axis`` because it provides more meaningful
        |  and dim_ordering-robust interface.

    Example
    -------
    A frequency-axis normalization after a spectrogram::

        model.add(Spectrogram())
        model.add(Normalization2D(stf_axis='freq))

    '''

    def __init__(self, str_axis=None, int_axis=None, image_data_format='default',
                 eps=1e-10, **kwargs):
        assert not (int_axis is None and str_axis is None), \
            'In Normalization2D, int_axis or str_axis should be specified.'

        assert image_data_format in ('channels_first', 'channels_last', 'default'), \
            'Incorrect image_data_format: {}'.format(image_data_format)

        if image_data_format == 'default':
            self.image_data_format = K.image_data_format()
        else:
            self.image_data_format = image_data_format

        if int_axis is None:
            assert str_axis in ('batch', 'data_sample', 'channel', 'freq', 'time'), \
                'Incorrect str_axis: %s' % str_axis
            if str_axis == 'batch':
                int_axis = -1
            elif str_axis == 'data_sample':
                int_axis = 0
            elif str_axis == 'channel':
                if self.image_data_format == 'channels_first':
                    int_axis = 1
                elif self.image_data_format == 'channels_last':
                    int_axis = 3
            elif str_axis == 'freq':
                if self.image_data_format == 'channels_first':
                    int_axis = 2
                elif self.image_data_format == 'channels_last':
                    int_axis = 1
            elif str_axis == 'time':
                if self.image_data_format == 'channels_first':
                    int_axis = 3
                elif self.image_data_format == 'channels_last':
                    int_axis = 2

        assert int_axis in (-1, 0, 1, 2, 3), 'invalid int_axis: ' + str(int_axis)
        self.axis = int_axis
        self.eps = eps
        super(Normalization2D, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if self.axis == -1:
            mean = K.mean(x, axis=[3, 2, 1, 0], keepdims=True)
            std = K.std(x, axis=[3, 2, 1, 0], keepdims=True)
        elif self.axis in (0, 1, 2, 3):
            all_dims = [0, 1, 2, 3]
            del all_dims[self.axis]
            mean = K.mean(x, axis=all_dims, keepdims=True)
            std = K.std(x, axis=all_dims, keepdims=True)
        return (x - mean) / (std + self.eps)

    def get_config(self):
        config = {'int_axis': self.axis,
                  'image_data_format': self.image_data_format}
        base_config = super(Normalization2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FrequencyWeighting(Layer):
    """

    The computation code is from librosa.
    Brian, thanks x NaN!

    """

    def __init__(self, mode, frequencies, decibel, power, **kwargs):
        """
        mode: 'A' 'a' for A-weighting
        frequencies: list of float or 1d np array, center frequencies.
        decibel: Bool, true if input is decibel scale (log(X))
        power: float but probably either 1.0 or 2.0.
        E.g., if input is power spectrogram which is log(X**2),
            power = 2.0,
            decibel = True
        If decibel:
            in call(), weights are ADDED.
        else:
            in call(), weights are MULTIPLIED.


        """
        # TODO: Current code is for keras v1.
        assert mode.lower() in ('a')
        self.mode = mode
        self.frequencies = frequencies
        self.decibel = decibel
        self.power = power
        super(FrequencyWeighting, self).__init__(**kwargs)

        freq_weights = backend.a_weighting(self.frequencies)

        if power != 2.0:
            freq_weights *= (power / 2.0)
        if decibel:
            self.freq_weights = K.variable(freq_weights, dtype=K.floatx())
        else:
            self.freq_weights = K.variable(10. ** freq_weights, dtype=K.floatx())

    def call(self, x, mask=None):
        if self.decibel:
            return x + self.freq_weights
        else:
            return x * self.freq_weights

    def get_config(self):
        config = {'mode': self.mode,
                  'frequencies': self.frequencies,
                  'decibel': self.decibel}
        base_config = super(FrequencyWeighting, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

