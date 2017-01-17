# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
from keras import backend as K
from keras.engine import Layer
from keras.utils.np_utils import conv_output_length
from . import backend


class Stft(Layer):
    '''Returns spectrogram(s) in 2D image format.
    
    # Arguments
        * n_dft: integer > 0 (scalar), power of 2. 
            number of DFT points

        * n_hop: integer > 0 (scalar), hop length

        * border_mode: string, `'same'` or `'valid'`

        * power: float (scalar), `2.0` if power-spectrogram,
            `1.0` if amplitude spectrogram

        * return_decibel_spectrogram: bool, returns decibel, 
            i.e. log10(amplitude spectrogram) if `True`

        * trainable_kernel: bool, set if the kernels are trainable

        * dim_ordering: string, `'th'` or `'tf'`.
            The returned spectrogram follows this dim_ordering convention.
            If `'default'`, follows the current Keras session's setting.

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
            # dim_ordering == 'th'
            from kapre.TimeFrequency import Spectrogram
            src = np.random.random((2, 44100))
            sr = 44100
            model = Sequential()
            model.add(Spectrogram(n_dft=512, n_hop=256, input_shape=src.shape, 
                      return_decibel=True, power_spectrogram=2.0, trainable_kernel=False,
                      name='static_stft'))
            model.summary(line_length=80, positions=[.33, .65, .8, 1.])

            # ________________________________________________________________________________
            # Layer (type)              Output Shape              Param #     Connected to    
            # ================================================================================
            # static_stft (Spectrogram) (None, 2, 257, 173)       0           spectrogram_inpu
            # ================================================================================
            # Total params: 0
            # ________________________________________________________________________________
        ```
        ```python
            model = Sequential()
            model.add(Spectrogram(n_dft=512, n_hop=256, input_shape=src.shape, 
                      return_decibel=True, power=2.0, trainable_kernel=True,
                      name='trainable_stft'))
            model.summary(line_length=80, positions=[.33, .6, .8, 1.])

            # ________________________________________________________________________________
            # Layer (type)              Output Shape          Param #         Connected to    
            # ================================================================================
            # trainable_stft (Spectrogr (None, 2, 257, 173)   263168          spectrogram_inpu
            # ================================================================================
            # Total params: 263168
            # ________________________________________________________________________________
            print(model.layers[0].trainable_weights)
            # [<TensorType(float32, 4D)>, <TensorType(float32, 4D)>]          
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

        self.n_dft = n_dft
        self.n_filter = (n_dft / 2) + 1
        self.trainable_kernel = trainable_kernel
        self.n_hop = n_hop
        self.border_mode = 'same'
        self.power_spectrogram = float(power_spectrogram)
        self.return_decibel_spectrogram = return_decibel_spectrogram
        self.dim_ordering = dim_ordering
        super(Spectrogram, self).__init__(**kwargs)

    def build(self, input_shape):
        '''input_shape: (n_ch, length)'''
        self.n_ch = input_shape[1]
        self.len_src = input_shape[2]
        self.is_mono = (self.n_ch == 1)
        if self.dim_ordering == 'th':
            self.ch_axis_idx = 1
        else:
            self.ch_axis_idx = 3
        assert self.len_src >= self.n_dft, 'Hey! The input is too short!'

        self.n_frame = conv_output_length(self.len_src,
                                    self.n_dft,
                                    self.border_mode,
                                    self.n_hop)

        dft_real_kernels, dft_imag_kernels = backend.get_stft_kernels(self.n_dft)
        self.dft_real_kernels = K.variable(dft_real_kernels)
        self.dft_imag_kernels = K.variable(dft_imag_kernels)
        # kernels shapes: (filter_length, 1, input_dim, nb_filter)?
        if self.trainable_kernel:
            self.trainable_weights.append(self.dft_real_kernels) 
            self.trainable_weights.append(self.dft_imag_kernels)
        else:
            self.non_trainable_weights.append(self.dft_real_kernels) 
            self.non_trainable_weights.append(self.dft_imag_kernels)

        self.built = True

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            return (input_shape[0], self.n_ch, self.n_filter, self.n_frame)
        else:
            return (input_shape[0], self.n_filter, self.n_frame, self.n_ch)

    def call(self, x, mask=None):
        '''computes spectrorgram ** power.'''
        output = self._spectrogram_mono(x[:, 0:1, :])
        if self.is_mono is False:
            for ch_idx in range(1, self.n_ch):
                output = K.concatenate((output, 
                           self._spectrogram_mono(x[:, ch_idx:ch_idx+1, :])),
                           axis=self.ch_axis_idx)
        if self.power_spectrogram != 2.0:
            output = K.pow(K.sqrt(output), self.power_spectrogram)
        if self.return_decibel_spectrogram:
            output = backend.amplitude_to_decibel(output)
        return output

    def get_config(self):
        config = {'n_dft': self.n_dft,
                  'n_hop': self.n_hop,
                  'border_mode': self.border_mode,
                  'power_spectrogram': self.power_spectrogram,
                  'return_decibel_spectrogram': self.return_decibel_spectrogram,
                  'trainable_kernel': self.trainable_kernel,
                  'dim_ordering':self.dim_ordering}
        base_config = super(Spectrogram, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _spectrogram_mono(self, x):
        '''x.shape : (None, 1, len_src),
        returns 2D batch of a mono power-spectrogram'''
        x = K.permute_dimensions(x, [0, 2, 1])
        x = K.expand_dims(x, 3)  # add a dummy dimension (channel axis)
        subsample = (self.n_hop, 1)
        output_real = K.conv2d(x, self.dft_real_kernels,
                               strides=subsample,
                               border_mode=self.border_mode,
                               dim_ordering='tf')
        output_imag = K.conv2d(x, self.dft_imag_kernels,
                               strides=subsample,
                               border_mode=self.border_mode,
                               dim_ordering='tf')
        output = output_real ** 2 + output_imag ** 2
        # now shape is (batch_sample, n_frame, 1, freq)
        if self.dim_ordering == 'tf':
            output = K.permute_dimensions(output, [0, 3, 1, 2])
        else:
            output = K.permute_dimensions(output, [0, 2, 3, 1])
        return output


