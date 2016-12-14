# -*- coding: utf-8 -*-
from __future__ import absolute_import
from keras.layers.noise import GaussianNoise
from keras import backend as K
import numpy as np
from keras.engine import Layer


class RandomMixer(Layer):
    '''Randomly mix a layer

    # Parameter
        layer: the layer in interest. `output_shape` of `layer` == `input_shape`.

        mix_mode: string, either 'soft' or 'hard'

        prob: 0. <= float <= 1., probability

        max_gain: float > 0., maximum gain

    # Example
    ```python
        wn = WhiteNoise()
        model.add(RandomMixer(wn, 'hard', prob=0.5))
    ```
    '''
    def __init__(self, layer, mix_mode, prob=None, max_gain=None):
        assert mix_mode in ('soft', 'hard')
        assert (prob is not None) or (max_gain is not None)

        if prob is not None:
            assert prob <= 1.0
            assert prob > 0.
            self.prob = prob

        if max_gain is not None:
            max_gain > 0.0
            self.max_gain = max_gain

        self.layer = layer
        self.mix_mode = mix_mode
        super(RandomMixer, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if self.mix_mode == 'soft':
            effect = self.layer.call(x, mask=mask)
            random_gain = np.random.uniform(low=0., high=self.max_gain)
            return x + random_gain * effect
        elif self.mix_mode == 'hard':
            if np.random.binomial(n=1, p=self.prob) == 1:
                return x + self.layer.call(x, mask=mask)
            else:
                return x

    def get_config(self):
        config = {}
        base_config = super(RandomMixer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WhiteNoise(GaussianNoise):
    '''Wrapper for keras.layers.noise.GaussianNoise'''
    def __init__(self, gain, sigma):
        self.gain = gain
        super(WhiteNoise, self).__init__(sigma, **kwargs)

    def get_config(self):
        config = {'gain':self.gain}
        base_config = super(WhiteNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




# dynamic range compression
# TODO
class DynamicRangeCompression1D(Layer):
    def __init__(self):
        pass




