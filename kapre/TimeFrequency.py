# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
from keras import backend as K
from keras.engine import Layer
from keras.utils.np_utils import conv_output_length
from . import backend


class Spectrogram(Layer):
    '''Returns spectrogram(s) in 2D image format.

    # Arguments
        * n_dft: integer > 0 (scalar), power of 2. 
            number of DFT points

        * n_hop: integer > 0 (scalar), hop length

        * border_mode: string, `'same'` or `'valid'`

        * power: float (scalar), `2.0` if power-spectrogram,
            `1.0` if amplitude spectrogram

        * return_decibel: bool, returns decibel, 
            i.e. log10(amplitude spectrogram) if `True`

        * trainable: bool, set if the kernels are trainable

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
                      return_decibel=True, power=2.0, trainable=False,
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
                      return_decibel=True, power=2.0, trainable=True,
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
    def __init__(self, n_dft, n_hop=None, border_mode='same', 
                 power=1.0, return_decibel=False,
                 trainable_kernel=False, dim_ordering='default', **kwargs):
        assert n_dft > 1 and ((n_dft & (n_dft - 1)) == 0), \
            ('n_dft should be > 1 and power of 2, but n_dft == %d' % n_dft)
        assert isinstance(trainable, bool)
        assert isinstance(return_decibel, bool)
        assert border_mode in ('same', 'valid')
        if n_hop is None:
            n_hop = n_dft / 2

        if dim_ordering == 'default':
            self.dim_ordering = K.image_dim_ordering()
        self.dim_ordering in ('th', 'tf')

        self.n_dft = n_dft
        self.n_filter = (n_dft / 2) + 1
        self.trainable_kernel = trainable_kernel
        self.n_hop = n_hop
        self.border_mode = 'same'
        self.power = float(power)
        self.return_decibel_spectrogram = return_decibel
        super(Spectrogram, self).__init__(trainable=trainable, **kwargs)

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
            self.trainable_weights.append(self.dft_imag_kernels])
        else:
            self.non_trainable_weights.append(self.dft_real_kernels) 
            self.non_trainable_weights.append(self.dft_imag_kernels])

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
        if self.power != 2.0:
            output = K.pow(K.sqrt(output), self.power)
        if self.return_decibel_spectrogram:
            output = backend.amplitude_to_decibel(output)
        return output

    def get_config(self):
        config = {'n_dft': self.n_dft,
                  'n_filter': self.n_filter,
                  'trainable': self.trainable,
                  'n_hop': self.n_hop,
                  'border_mode': self.border_mode,
                  'power': self.power,
                  'return_decibel_spectrogram': self.return_decibel_spectrogram}
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


class Melspectrogram(Spectrogram):
    '''Returns mel-spectrogram(s) in 2D image format.

    # Arguments
        * n_dft: integer > 0 (scalar), power of 2. 
            number of DFT points

        * n_hop: integer > 0 (scalar), hop length

        * border_mode: string, `'same'` or `'valid'`

        * sr: integer > 0 (scalar), sampling rate

        * n_mels: integer > 0 (scalar), number of mel bands

        * fmin: float > 0 (scalar), minimum frequency to include in melgram

        * fmax: float > fmin (scalar), maximum frequency to include in melgram

        * power: float (scalar), `2.0` if power-spectrogram,
            `1.0` if amplitude spectrogram

        * return_decibel: bool, returns decibel, 
            i.e. log10(amplitude spectrogram) if `True`

        * trainable: bool, set if the kernels are trainable

        * dim_ordering: string, `'th'` or `'tf'`.
            The returned spectrogram follows this dim_ordering convention.
            If `'default'`, follows the current Keras session's setting.

    # Input shape
        * 2D array, `(audio_channel, audio_length)`.
            E.g., `(1, 44100)` for mono signal,
                `(2, 44100)` for stereo signal.
            It supports multichannel signal input.

    # Returns
        * abs(mel-spectrogram) in a shape of 2D data, i.e.,
            `(None, n_channel, n_mels, n_time)` if `'th'`,
            `(None, n_mels, n_time, n_channel)` if `'tf'`,

    # Example
        ```python
            dim_ordering == 'th'
            from kapre.TimeFrequency import Melspectrogram
            src = np.random.random((2, 44100))
            sr = 44100
            model = Sequential()
            model.summary()
            # ________________________________________________________________________________
            # Layer (type)              Output Shape          Param #         Connected to    
            # ================================================================================
            # static_melgram (Melspectr (None, 2, 128, 173)   0               melspectrogram_i
            # ================================================================================
            # Total params: 0
            # ________________________________________________________________________________
        ```
        ```python
            model = Sequential()
            model.add(Melspectrogram(n_dft=512, n_hop=256, input_shape=src.shape,
                                     sr=sr, n_mels=128, fmin=0.0, fmax=16000,
                                     power=2.0, return_decibel=True,
                                     trainable=True,
                                     name='trainable_melgram'))
            model.summary(line_length=80, positions=[.33, .6, .8, 1.])
            # ________________________________________________________________________________
            # Layer (type)              Output Shape          Param #         Connected to    
            # ================================================================================
            # trainable_melgram (Melspe (None, 2, 128, 173)   296064          melspectrogram_i
            # ================================================================================
            # Total params: 296064
            # ________________________________________________________________________________
            print(model.layers[0].trainable_weights)
            # [<TensorType(float32, 4D)>, <TensorType(float32, 4D)>, <TensorType(float32, matrix)>]
            # real kernels, imaginary kernels, and freq-to-mel matrix
            
        ```
    '''    
    def __init__(self, n_dft, n_hop=None, border_mode='same',
                 sr=22050, n_mels=128, fmin=0.0, fmax=None, 
                 power=1.0, return_decibel=False,
                 trainable_fb=False, trainable_kernel=False, dim_ordering='default', **kwargs):
        super(Melspectrogram, self).__init__(n_dft, n_hop, border_mode,
                                             2.0, False,
                                             trainable_kernel, dim_ordering, **kwargs)
        assert sr > 0
        assert fmin >= 0.0
        if fmax is None:
            fmax = float(sr) / 2
        assert fmax > fmin
        assert isinstance(return_decibel, bool)

        self.sr = int(sr)
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.return_decibel_melgram = return_decibel
        self.trainable_fb = trainable_fb

    def build(self, input_shape):
        super(Melspectrogram, self).build(input_shape)
        self.built = False
        # compute freq2mel matrix
        mel_basis = backend.mel(self.sr, self.n_dft, self.n_mels, self.fmin, self.fmax)  # (128, 1025) (mel_bin, n_freq)
        mel_basis = np.transpose(mel_basis)
        
        self.freq2mel = K.variable(mel_basis)
        if self.trainable_fb:
            self.trainable_weights.append(self.freq2mel)
        else:
            self.non_trainable_weights.append(self.freq2mel)
        self.built = True

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            return (input_shape[0], self.n_ch, self.n_mels, self.n_frame)
        else:
            return (input_shape[0], self.n_mels, self.n_frame, self.n_ch)

    def call(self, x, mask=None):
        power_spectrogram = super(Melspectrogram, self).call(x, mask)
        # now,  th: (batch_sample, n_ch, n_freq, n_time)
        #       tf: (batch_sample, n_freq, n_time, n_ch)
        if self.dim_ordering == 'th':
            power_spectrogram = K.permute_dimensions(power_spectrogram, [0, 1, 3, 2])
        else:
            power_spectrogram = K.permute_dimensions(power_spectrogram, [0, 3, 2, 1])
        # now, whatever dim_ordering, (batch_sample, n_ch, n_time, n_freq)
        output = K.dot(power_spectrogram, self.freq2mel) 
        if self.dim_ordering == 'th':
            output = K.permute_dimensions(output, [0, 1, 3, 2])
        else:
            output = K.permute_dimensions(output, [0, 3, 2, 1])
        if self.power != 2.0:
            output = K.pow(K.sqrt(output), self.power)
        if self.return_decibel_melgram:
            output = backend.amplitude_to_decibel(output)
        return output

    def get_config(self):
        config = {'sr': self.sr,
                  'n_mels': self.n_mels,
                  'fmin': self.fmin,
                  'fmax': self.fmax,
                  'return_decibel_melgram': self.return_decibel_melgram}
        base_config = super(Melspectrogram, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
