# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
from keras import backend as K
from keras.engine import Layer, InputSpec


class AdditiveNoise(Layer):
    """
    ### `AdditiveNoise`

    ```python
    kapre.augmentation.AdditiveNoise(power=0.1, random_gain=False, noise_type='white', **kwargs)
    ```
    Add noise to input data and output it.

    #### Parameters
    
    * power: float [scalar]
        - The power of noise. std if it's white noise.
        - Default: ``0.1``

    * random_gain: bool
        - Whether the noise gain is random or not.
        - If ``True``, gain is sampled from ``uniform(low=0.0, high=power)`` in every batch.
        - Default: ``False``

    * noise_type; str,
        - Specify the type of noise. It only supports ``'white'`` now.
        - Default: ```white```


    #### Returns

    Same shape as input data but with additional generated noise.

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
