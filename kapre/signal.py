"""Signal layers"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
from . import backend
from tensorflow.keras import backend as K
from .backend import CH_FIRST_STR, CH_LAST_STR, CH_DEFAULT_STR


class Frame(Layer):
    def __init__(
        self, frame_length, hop_length, pad_end=False, pad_value=0, data_format='default', **kwargs
    ):
        """

        Args:
            frame_length (int):
            hop_length (int):
            pad_end (bool):
            pad_value (int or float):
            data_format (str): 'channels_first', 'channels_last', or `default`
            **kwargs:
        """
        super(Frame, self).__init__(**kwargs)

        backend.validate_data_format_str(data_format)

        self.frame_length = frame_length
        self.hop_length = hop_length
        self.pad_end = pad_end
        self.pad_value = pad_value

        if data_format == CH_DEFAULT_STR:
            self.data_format = K.image_data_format()
        else:
            self.data_format = data_format

        if data_format == CH_FIRST_STR:
            self.time_axis = 2  # batch, ch, time
        else:
            self.time_axis = 1  # batch, time, ch

    def call(self, x):
        """
        Frame the input signal using `tf.signal.frame`.

        Args:
            x (`Tensor`): batch audio signal in the specified 1D format in initiation.

        Returns:
            (`Tensor`): A framed tensor. The shape is
                (batch, time (frames), frame_length, channel) if `channels_last`,
                (batch, channel, time (frames), frame_length) if `channels_first`.
        """
        return tf.signal.frame(
            x,
            frame_length=self.frame_length,
            frame_step=self.hop_length,
            pad_end=self.pad_end,
            pad_value=self.pad_value,
            axis=self.time_axis,
        )
