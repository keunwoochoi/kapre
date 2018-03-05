# -*- coding: utf-8 -*-
from __future__ import absolute_import
from keras.engine import Layer
from keras import backend as K
from kapre import backend


# Todo: Filterbank(); init with mel, log, linear, etc.
# not parameterised, just a matrix multiplication

class Filterbank(Layer):
    """
    ### `Filterbank`

    `kapre.filterbank.Filterbank(n_fbs, trainable_fb, sr=None, init='mel', fmin=0., fmax=None,
                                 bins_per_octave=12, image_data_format='default', **kwargs)`

    #### Notes
        Input/output are 2D image format.
        E.g., if channel_first,
            - input_shape: ``(None, n_ch, n_freqs, n_time)``
            - output_shape: ``(None, n_ch, n_mels, n_time)``


    #### Parameters
    * n_fbs: int
       - Number of filterbanks

    * sr: int
        - sampling rate. It is used to initialize ``freq_to_mel``.

    * init: str
        - if ``'mel'``, init with mel center frequencies and stds.

    * fmin: float
        - min frequency of filterbanks.
        - If `init == 'log'`, fmin should be > 0. Use `None` if you got no idea.

    * fmax: float
        - max frequency of filterbanks.
        - If `init == 'log'`, fmax is ignored.

    * trainable_fb: bool,
        - Whether the filterbanks are trainable or not.

    """

    def __init__(self, n_fbs, trainable_fb, sr=None, init='mel', fmin=0., fmax=None,
                 bins_per_octave=12, image_data_format='default', **kwargs):
        """ TODO: is sr necessary? is fmax necessary? init with None?  """
        self.supports_masking = True
        self.n_fbs = n_fbs
        assert init in ('mel', 'log', 'linear', 'uni_random')
        if fmax is None:
            self.fmax = sr / 2.
        else:
            self.fmax = fmax
        if init in ('mel', 'log'):
            assert sr is not None

        self.fmin = fmin
        self.init = init
        self.bins_per_octave = bins_per_octave
        self.sr = sr
        self.trainable_fb = trainable_fb
        assert image_data_format in ('default', 'channels_first', 'channels_last')
        if image_data_format == 'default':
            self.image_data_format = K.image_data_format()
        else:
            self.image_data_format = image_data_format
        super(Filterbank, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.image_data_format == 'channels_first':
            self.n_ch = input_shape[1]
            self.n_freq = input_shape[2]
            self.n_time = input_shape[3]
        else:
            self.n_ch = input_shape[3]
            self.n_freq = input_shape[1]
            self.n_time = input_shape[2]

        if self.init == 'mel':
            self.filterbank = K.variable(backend.filterbank_mel(sr=self.sr,
                                                                n_freq=self.n_freq,
                                                                n_mels=self.n_fbs,
                                                                fmin=self.fmin,
                                                                fmax=self.fmax).transpose(),
                                         dtype=K.floatx())
        elif self.init == 'log':
            self.filterbank = K.variable(backend.filterbank_log(sr=self.sr,
                                                                n_freq=self.n_freq,
                                                                n_bins=self.n_fbs,
                                                                bins_per_octave=self.bins_per_octave,
                                                                fmin=self.fmin).transpose(),
                                         dtype=K.floatx())

        if self.trainable_fb:
            self.trainable_weights.append(self.filterbank)
        else:
            self.non_trainable_weights.append(self.filterbank)
        super(Filterbank, self).build(input_shape)
        self.built = True

    def compute_output_shape(self, input_shape):
        if self.image_data_format == 'channels_first':
            return input_shape[0], self.n_ch, self.n_fbs, self.n_time
        else:
            return input_shape[0], self.n_fbs, self.n_time, self.n_ch

    def call(self, x):
        # reshape so that the last axis is freq axis
        if self.image_data_format == 'channels_first':
            x = K.permute_dimensions(x, [0, 1, 3, 2])
        else:
            x = K.permute_dimensions(x, [0, 3, 2, 1])
        output = K.dot(x, self.filterbank)
        # reshape back
        if self.image_data_format == 'channels_first':
            return K.permute_dimensions(output, [0, 1, 3, 2])
        else:
            return K.permute_dimensions(output, [0, 3, 2, 1])

    def get_config(self):
        config = {'n_fbs': self.n_fbs,
                  'sr': self.sr,
                  'init': self.init,
                  'fmin': self.fmin,
                  'fmax': self.fmax,
                  'bins_per_octave': self.bins_per_octave,
                  'trainable_fb': self.trainable_fb}
        base_config = super(Filterbank, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
