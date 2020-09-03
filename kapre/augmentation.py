import tensorflow as tf
from tensorflow.keras import backend as K
from . import backend
from .backend import _CH_FIRST_STR, _CH_LAST_STR, _CH_DEFAULT_STR
from tensorflow.python.ops import array_ops
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import tf_utils

__all__ = [
    'SpecAugment'
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