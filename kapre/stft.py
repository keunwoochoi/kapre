# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
from keras import backend as K
from keras.engine import Layer
from keras.utils.np_utils import conv_output_length
from . import backend
from . import backend_keras

from theano import tensor as T
from theano.tensor import fft


class Stft(Layer):
    '''Returns STFT in 2D image format. It uses Fast Fourier transform (FFT)
    Now only support for theano.
    
    # Arguments
        * `n_fft`: integer > 0 (scalar), power of 2. 
            number of DFT points
            Default: `512`

        * `n_hop`: integer > 0 (scalar), hop length.
            If `None`, `n_fft / 2` is used.
            Default: `None`

        * `power_stft`: float (scalar), `2.0` if power-spectrogram,
            `1.0` if amplitude spectrogram
            Default: `2.0`

        * `return_decibel_stft`: bool, returns decibel, 
            i.e. log10(amplitude spectrogram) if `True`
            Default: `False`

        * `dim_ordering`: string, `'th'` or `'tf'`.
            The returned spectrogram follows this dim_ordering convention.
            If `'default'`, follows the current Keras session's setting.
            Default: `'default'`

    # Input shape
        * 2D array, `(audio_channel, audio_length)`.
            E.g., `(1, 44100)` for mono signal,
                `(2, 44100)` for stereo signal.
            It supports multichannel signal input.

    # Returns
        * abs(Spectrogram) in a shape of 2D data, i.e.,
            `(None, n_channel, n_freq, n_time)` if `'th'`,
            `(None, n_freq, n_time, n_channel)` if `'tf'`,

    # Example
        ```python
        import keras
        import kapre
        from keras.models import Sequential
        from kapre.stft import Stft
        import numpy as np

        print('Keras version: {}'.format(keras.__version__))
        print('Keras backend: {}'.format(keras.backend._backend))
        print('Keras image dim ordering: {}'.format(keras.backend.image_dim_ordering()))
        print('Kapre version: {}'.format(kapre.__version__))

        src = np.random.random((2, 44100))
        sr = 44100
        model = Sequential()
        model.add(Stft(n_fft=256, n_hop=64, return_decibel_stft=True,
                      input_shape=src.shape))
        model.summary(line_length=80, positions=[.33, .65, .8, 1.])

        # Keras version: 1.2.1
        # Keras backend: theano
        # Keras image dim ordering: th
        # Kapre version: 0.0.3
        # ________________________________________________________________________________
        # Layer (type)              Output Shape              Param #     Connected to    
        # ================================================================================
        # stft_4 (Stft)             (None, 129, 686, 2)       0           stft_input_4[0][
        # ================================================================================
        # Total params: 0
        # Trainable params: 0
        # Non-trainable params: 0
        # ________________________________________________________________________________

        ```
    '''
    def __init__(self, n_fft=512, n_hop=None,
                 power_stft=2.0, return_decibel_stft=False,
                 dim_ordering='default', **kwargs):
        assert n_fft > 1 and ((n_fft & (n_fft - 1)) == 0), \
            ('n_fft should be > 1 and power of 2, but n_fft == %d' % n_fft)
        assert isinstance(return_decibel_stft, bool)
        # assert border_mode in ('same', 'valid')
        if n_hop is None:
            n_hop = n_dft / 2

        assert dim_ordering in ('default', 'th', 'tf')

        if dim_ordering == 'default':
            self.dim_ordering = K.image_dim_ordering()
        else:
            self.dim_ordering = dim_ordering

        self.n_fft = n_fft
        self.n_freq = (n_fft / 2) + 1
        self.n_hop = n_hop
        # self.border_mode = 'same'
        self.power_stft = float(power_stft)
        self.return_decibel_stft = return_decibel_stft
        self.dim_ordering = dim_ordering
        super(Stft, self).__init__(**kwargs)

    def build(self, input_shape):
        '''input_shape: (n_ch, length)'''
        self.n_ch = input_shape[1]
        self.len_src = input_shape[2]
        self.is_mono = (self.n_ch == 1)
        if self.dim_ordering == 'th':
            self.ch_axis_idx = 1
        else:
            self.ch_axis_idx = 3
        assert self.len_src >= self.n_fft, 'Hey! The input is too short!'
        self.n_frame = conv_output_length(self.len_src,
                                    self.n_fft,
                                    'valid',
                                    self.n_hop)
        self.fft_window = backend._hann(self.n_fft, sym=False)
        self.built = True

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            return (input_shape[0], self.n_ch, self.n_freq, self.n_frame)
        else:
            return (input_shape[0], self.n_freq, self.n_frame, self.n_ch)

    def call(self, x, mask=None):
        '''computes stft ** power.'''
        for fr_idx in range(self.n_frame):    
            X_frame_power = K.sum(K.square(fft.rfft(
                                        self.fft_window * x[:, :, fr_idx * self.n_hop :
                                                           fr_idx * self.n_hop + self.n_fft]
                                    )), axis=3, keepdims=True)
            if fr_idx == 0:
                output = X_frame_power
            else:
                output = T.concatenate([output, X_frame_power], axis=3)

        if self.power_stft != 2.0:
            output = K.pow(output, self.power_stft/2.0)
        if self.return_decibel_stft:
            output = backend_keras.amplitude_to_decibel(output)
        return output

    def get_config(self):
        config = {'n_fft': self.n_fft,
                  'n_hop': self.n_hop,
                  'power_stft': self.power_stft,
                  'return_decibel_stft': self.return_decibel_stft,
                  'dim_ordering':self.dim_ordering}
        base_config = super(Spectrogram, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
