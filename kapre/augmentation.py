"""Augmentation layers.

This module includes augmentation layers that can be applied to audio data and representations.

"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from . import backend
from .backend import _CH_FIRST_STR, _CH_LAST_STR, _CH_DEFAULT_STR
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.utils import tf_utils

__all__ = [
    'SpecAugment',
    'ChannelSwap'
]

class SpecAugment(Layer):
    """
    A layer to mask time and frequency 
    See  `ARXIV <https://arxiv.org/abs/1904.08779>`_ for more details.

    Add masking to input data and output it

    Example:
        ::
            input_shape = (batch_size, freq, time, 1)  # Spectrogram
            model = Sequential()
            model.add(melspectrogram_layer)
            model.add(kapre.augmentation.SpecAugment(10,10,name='SpecAugment'))
  
    Args:
        x: Spectrogram
        freq_param:int Param of freq masking
        time_param:int Param of Time masking
    Returns: 
        Masked Tensor of Spectrogram
    """
    def __init__(
        self, 
        freq_param = None, 
        time_param = None,
        input_data_format:str='default',
        **kwargs
    ):
        
        super(SpecAugment, self).__init__(**kwargs)

        backend.validate_data_format_str(input_data_format)

        self.freq_param = freq_param
        self.time_param = time_param

        if not freq_param and not time_param:
            raise RuntimeError("at least one param value should be defined")
        
        idt = input_data_format
        self.input_data_format = K.image_data_format() if idt == _CH_DEFAULT_STR else idt

    def call(self, x, training=None):
        
        if training is None:
            training = K.learning_phase()

        ch_axis = 1 if self.input_data_format == 'channels_first' else 3
        
        if K.int_shape(x)[ch_axis] != 1:
            raise RuntimeError('SpecAugment does not support images with depth greater than 1')

        if self.freq_param is not None:
            x = tf_utils.smart_cond(training, 
                                     lambda: backend.random_masking_along_axis(x, param=self.freq_param, axis = 0),
                                     lambda: array_ops.identity(x))
        if self.time_param is not None:
            x = tf_utils.smart_cond(training, 
                                     lambda: backend.random_masking_along_axis(x, param=self.time_param, axis = 1),
                                     lambda: array_ops.identity(x))
        return x
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(SpecAugment, self).get_config()
        config.update(
            {
                'freq_param': self.freq_param,
                'time_param': self.time_param,
                'image_data_format': self.input_data_format,
            }
        )
        
        return config

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
      