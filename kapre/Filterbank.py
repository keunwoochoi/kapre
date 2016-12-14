# -*- coding: utf-8 -*-
from keras import initializations
from keras.engine import Layer
from keras import backend as K
import numpy as np
from TimeFrequency import _mel, _mel_frequencies


class ParametricMel(Layer):
    '''Parametric mel assumes a mono STFT input, i.e., (None, row, col).
    It converts 2d stft image into 2d filterbank-ed image.
    
    The conversion matrix, `freq_to_mel` in `call()` is parameterized by 
    `self.n_mels` Gaussian kernels, of which means and stds are initialized with
    `self.means_init` and `self.stds_init`. The means and stds are trainable.

    # Shapes
        input_shape: (None, n_freqs, n_time)

        output_shape: (None, n_mels, n_time)
    
    # Arguments
        n_mels: integer, number of mel-bins

        n_freqs: integer, should match to input data's n_freqs

        sr: integer, sampling rate, used to initialize freq_to_mel

        scale: float, sum of gaussian kernels. Default: 24. that kinda match the total energy.

        init: string. if 'mel', init with mel center frequencies and stds.
    
    '''
    def __init__(self, n_mels, n_freqs, sr, scale=24., init='mel', fmax=None, **kwargs):
        self.supports_masking = True
        self.scale = scale # scaling
        self.n_mels = n_mels
        assert init in ('mel', 'linear', 'log', 'uni_random')
        if fmax is None:
            fmax = sr / 2.
        if init == 'mel':
            self.means_init = np.array(_mel_frequencies(n_mels, fmin=0.0, 
                                                        fmax=sr/2), 
                                       dtype='float32')
        elif init == 'linear':
            f_between = float(fmax) / n_mels
            f_low = f_between / 2.
            self.means_init = np.arange(f_low, fmax, f_between)
        # TODO
        # elif init == 'log':
        #     self.means_init = 
        # elif init == 'uni_random':
        #     self.means_init = 

        stds = self.means_init[1:] - self.means_init[:-1]
        self.stds_init = 0.3 * np.hstack((stds[0:1], stds[:])) # 0.3: kinda make sense by the resulting images..
        self.center_freqs_init = [float(i)*sr/2/(n_freqs-1) for i in range(n_freqs)] # dft frequencies

        super(ParametricMel, self).__init__(**kwargs)

    def build(self, input_shape):
        self.means = K.variable(self.means_init, 
                                name='{}_means'.format(self.name))
        self.stds =  K.variable(self.stds_init, 
                                name='{}_stds'.format(self.name))
        
        self.center_freqs_init = np.array(self.center_freqs_init)[np.newaxis, :] # (1, n_freq)
        self.center_freqs_init = np.tile(self.center_freqs_init, (self.n_mels, 1)) # (n_mels, n_freq)
        self.center_freqs = K.variable(self.center_freqs_init,
                                       name='{}_center_freqs'.format(self.name))
        self.trainable_weights = [self.means, self.stds] # [self.means, self.stds]
        self.n_freq = input_shape[1]
        self.n_time = input_shape[2]
        self.built = True

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.n_mels, input_shape[2])

    def call(self, x, mask=None):
        means = K.expand_dims(self.means, dim=1)
        stds = K.expand_dims(self.stds, dim=1)
        freq_to_mel = (self.scale * K.exp(-1. * K.square(self.center_freqs - means) \
                           / (2. * K.square(stds)))) \
                          / (np.sqrt(2. * np.pi).astype('float32') * stds)  # (n_mel, n_freq)
        out = K.dot(freq_to_mel, x) # (n_mel, None, n_time)
        return K.permute_dimensions(out, (1, 0, 2))

    def get_config(self):
        config = {'n_mels': self.n_mels}
        base_config = super(ParametricMel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

