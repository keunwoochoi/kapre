# -*- coding: utf-8 -*-
"""
Data augmentation
=================
"""
from __future__ import absolute_import
import numpy as np
from keras import backend as K
from keras.engine import Layer, InputSpec


class AdditiveNoise(Layer):
    """Add noise to input data and output it.

    Parameters
    ----------
    power: float [scalar]
        |  The power of noise. std if it's white noise.
        |  Default: ``0.1``

    random_gain: bool
        |  Whether the noise gain is random or not.
        |  If ``True``, gain is sampled from ``uniform(low=0.0, high=power)`` in every batch.
        |  Default: ``False``

    noise_type; str,
        |  Specify the type of noise. It only supports ``'white'`` now.
        |  Default: ```white```


    Returns
    -------
    Same as input data but with additional generated noise.

    Examples
    --------
    Adding a white noise on the top of Mel-spectrogram

        import keras
        import kapre
        from keras.models import Sequential
        from kapre.time_frequency import Melspectrogram
        from kapre.augmentation import AdditiveNoise
        import numpy as np

        print('Keras version: {}'.format(keras.__version__))
        print('Keras backend: {}'.format(keras.backend._backend))
        print('Keras image dim ordering: {}'.format(keras.backend.image_dim_ordering()))
        print('Kapre version: {}'.format(kapre.__version__))

        src = np.random.random((2, 44100))
        sr = 44100
        model = Sequential()
        model.add(Melspectrogram(sr=16000, n_mels=128, 
                  n_dft=512, n_hop=256, input_shape=src.shape, 
                  return_decibel_spectrogram=True,
                  trainable_kernel=False, name='melgram'))
        model.add(AdditiveNoise(power=0.2))
        model.summary(line_length=80, positions=[.33, .65, .8, 1.])

        # Keras version: 1.2.1
        # Keras backend: theano
        # Keras image dim ordering: th
        # Kapre version: 0.0.3
        # ________________________________________________________________________________
        # Layer (type)              Output Shape              Param #     Connected to    
        # ================================================================================
        # melgram (Melspectrogram)  (None, 2, 128, 173)       296064      melspectrogram_i
        # ________________________________________________________________________________
        # additivenoise_1 (Additive (None, 2, 128, 173)       0           melgram[0][0]   
        # ================================================================================
        # Total params: 296,064
        # Trainable params: 0
        # Non-trainable params: 296,064
        # ________________________________________________________________________________

    """

    def __init__(self, power=0.1, random_gain=False, noise_type='white', **kwargs):
        assert noise_type in ['white']
        self.supports_masking = True
        self.power = power
        self.random_gain = random_gain
        self.noise_type = noise_type
        self.uses_learning_phase = True
        super(AdditiveNoise, self).__init__(**kwargs)

    def call(self, x):
        if self.random_gain:
            noise_x = x + K.random_normal(shape=K.shape(x),
                                          mean=0.,
                                          stddev=np.random.uniform(0.0, self.power))
        else:
            noise_x = x + K.random_normal(shape=K.shape(x),
                                          mean=0.,
                                          stddev=self.power)

        return K.in_train_phase(noise_x, x)

    def get_config(self):
        config = {'power': self.power,
                  'random_gain': self.random_gain,
                  'noise_type': self.noise_type}
        base_config = super(AdditiveNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
