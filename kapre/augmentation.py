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
    # `SpecAugment`

    ```python
    kapre.augmentation.SpecAugment(time_param=10, freq_param=10, **kwargs)
    ```

    Add masking to input data and output it
    Args:
        x: Spectrogram
        freq_param: Param of freq masking
        time_param: Param of Time masking
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
        self.uses_learning_phase = True
        self.supports_masking = True

        assert (freq_param is not None) or (time_param is not None), "at least one param value should be defined"
        
        idt = input_data_format
        self.input_data_format = K.image_data_format() if idt == _CH_DEFAULT_STR else idt

    def call(self, x, training=None):
        
        if training is None:
            training = K.learning_phase()
            
        if self.input_data_format == 'channels_first':
            assert x.shape[1] == 1, 'SpecAugment does not support 2D images yet'

        else:
            assert x.shape[3] == 1, 'SpecAugment does not support 2D images yet'
        
        if self.freq_param is not None:
            x = tf_utils.smart_cond(training, 
                                     lambda: backend.freq_mask(x, param=self.freq_param),
                                     lambda: array_ops.identity(x))
        if self.time_param is not None:
            x = tf_utils.smart_cond(training, 
                                     lambda: backend.time_mask(x, param=self.time_param),
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