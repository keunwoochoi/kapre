"""Augmentation layers.

This module includes augmentation layers that can be applied to audio data and representations.

"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from . import backend
from .backend import _CH_FIRST_STR, _CH_LAST_STR, _CH_DEFAULT_STR
import numpy as np


class ChannelSwap(Layer):
    """
    Randomly swap the channel

    Args:
        data_format (`str`): specifies the data format of batch input/output
        **kwargs: Keyword args for the parent keras layer (e.g., `name`)

    Example:
        ::

            input_shape = (2048, 2)  # stereo signal
            n_fft = 1024
            n_hop = n_fft // 2
            kwargs = {
                'sample_rate': 22050,
                'n_freq': n_fft // 2 + 1,
                'n_mels': 128,
                'f_min': 0.0,
                'f_max': 8000,
            }
            model = Sequential()
            model.add(kapre.STFT(n_fft=n_fft, hop_length=n_hop, input_shape=input_shape))
            model.add(Magnitude())
            # (batch, n_frame=3, n_freq=n_fft // 2 + 1, ch=1) and dtype is float
            model.add(ChannelSwap())
            # same shape but with randomly swapped channels.

        ::

            input_shape = (2048, 4)  # mono signal
            model = Sequential()
            model.add(ChannelSwap(input_shape=input_shape))
            # still (2048, 4) but with randomly swapped channels

    """

    def __init__(
        self,
        data_format='default',
        **kwargs,
    ):
        backend.validate_data_format_str(data_format)

        if data_format == _CH_DEFAULT_STR:
            self.data_format = K.image_data_format()
        else:
            self.data_format = data_format

        super(ChannelSwap, self).__init__(**kwargs)

    def call(self, x, training=None):
        """
        Apply random channel-swap augmentation to `x`.

        Args:
            x (`Tensor`): A batch tensor of 1D (signals) or 2D (spectrograms) data
        """
        if training in (None, False):
            return x

        # figure out input data format
        if K.ndim(x) not in (3, 4):
            raise ValueError(
                'ndim of input tensor x should be 3 (batch signal) or 4 (batch spectrogram),'
                'but it is %d' % K.ndim(x)
            )

        if self.data_format == _CH_LAST_STR:
            ch_axis = 3 if K.ndim(x) == 4 else 2
        else:
            ch_axis = 1

        # get swap indices
        n_ch = K.int_shape(x)[ch_axis]
        if n_ch == 1:
            return x
        swap_indices = np.random.permutation(n_ch).tolist()

        # swap and return
        return tf.gather(x, indices=swap_indices, axis=ch_axis)

    def get_config(self):
        config = super(ChannelSwap, self).get_config()
        config.update(
            {
                'data_format': self.data_format,
            }
        )
        return config
