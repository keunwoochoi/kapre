# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
from keras import backend as K
from keras.engine import Layer


class AdditiveNoise(Layer):
    """Add noise to input

    # Arguments
        power: float (scalar), the power of noise. std if it's white noise.
            default: 0.1

        random_gain: bool, if the gain would be random.
            If true, uniform(low=0.0, high=power) is multiplied.

        noise_type; str, now only support 'white'.

    # Input shapes
        doesn't matter.

    # Returns
        input + generated noise.

    # Examples

    """
    def __init__(self, power=0.1, random_gain=False, noise_type='white', **kwargs):
        self.supports_masking = True
        self.power = power
        self.random_gain = random_gain
        self.noise_type = noise_type
        super(AdditiveNoise, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if self.random_gain:
            x + K.random_normal(shape=K.shape(x),
                                       mean=0.,
                                       std=np.random.uniform(0.0, self.power))
        else:
            return x + K.random_normal(shape=K.shape(x),
                                       mean=0.,
                                       std=self.power)

    def get_config(self):
        config = {'power': self.power,
                  'random_gain': self.random_gain
                  'noise_type': self.type}
        base_config = super(AdditiveNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
