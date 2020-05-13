# Oscar Tokunaga: oscar.tokunaga@hotmail.com

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class ComputeDeltas(Layer):
    """
    ### Deltas

    ```python
    kapre.delta.ComputeDeltas(win_length, mode, **kwargs)
    ```
    Calculate Deltas.

    #### Parameters
    
    * win_length: int
        - Window length.
        - Default: 5

    * mode: str
        - pad parameter
        - Default: 'SYMMETRIC'


    #### Returns

    Same shape as input data.

    """

    def __init__(self, 
                 win_length:int=5, 
                 mode:str='SYMMETRIC', 
                 data_format:str='default',
                 **kwargs):
        
        assert mode in ['SYMMETRIC','REFLECT','CONSTANT']
        
        assert data_format in ('default', 'channels_first', 'channels_last')
        if data_format == 'default':
            self.data_format = K.image_data_format()
        else:
            self.data_format = data_format

        self.win_length = win_length
        self.mode = mode
        super(ComputeDeltas, self).__init__(**kwargs)

    def call(self, x):
        
        assert self.win_length >= 3
        
        n = (self.win_length - 1) // 2
        denom = n * (n + 1) * (2 * n + 1) / 3
        
        if self.data_format=='channels_first':
            x = K.permute_dimensions(x,(0,2,3,1))
            
        x = tf.pad(x,tf.constant([[0,0],[0,0],[n,n],[0,0]]), mode="SYMMETRIC")
        kernel = K.arange(-n,n+1,1,dtype=K.floatx())
        kernel = K.reshape(kernel,(1,kernel.shape[-1],1,1))
        
        x = K.conv2d(x,kernel,1,data_format='channels_last')/denom
        
        if self.data_format=='channels_first':
            x = K.permute_dimensions(x,(0,3,1,2))
            
        return x

    def get_config(self):
        config = {'win_length': self.win_length,
                  'mode': self.mode}
        base_config = super(ComputeDeltas, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

