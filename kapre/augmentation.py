"""Augmentation layers.

This module includes augmentation layers that can be applied to audio data and representations.

"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from . import backend
from .backend import _CH_FIRST_STR, _CH_LAST_STR, _CH_DEFAULT_STR
import numpy as np

__all__ = ['SpecAugment', 'ChannelSwap']


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


class SpecAugment(Layer):
    """
    Apply SpecAugment to a Spectrogram. For more info, check the original paper at:
    https://arxiv.org/abs/1904.08779

    Args:
        freq_mask_param (`int`): Frequency Mask Parameter (F in the paper)
        time_mask_param (ìnt`): Time Mask Parameter (T in the paper)
        n_freq_masks (`int`): Number of frequency masks to apply (mF in the paper). By default is 1.
        n_time_masks (`int`): Number of time masks to apply (mT in the paper). By default is 1.
        mask_value (`float`): Value of the applied masks. By default is 0.
        data_format (`str`): specifies the data format of batch input/output
        **kwargs: Keyword args for the parent keras layer (e.g., `name`)

    Example:
        ::

            input_shape = (2048, 1)  # mono signal

            # We compute the Mel Spectrogram of the input signal
            melgram = kapre.composed.get_melspectrogram_layer(input_shape=input_shape,
                                                  n_fft=1024,
                                                  return_decibel=True,
                                                  n_mels=256,
                                                  input_data_format='channels_last',
                                                  output_data_format='channels_last')


            # Now we define the SpecAugment layer. It will apply 5 masks in the frequency axis,
            # 3 masks in the time axis. The frequency mask param is 5 and the time mask param
            # is 10.
            spec_augment = SpecAugment(freq_mask_param=5,
                                       time_mask_param=10,
                                       n_freq_masks=5,
                                       n_time_masks=3)

            model = Sequential()
            model.add(melgram)

            # Add the spec_augment layer for augmentation
            model.add(spec_augment)
        ::
    """

    def __init__(
        self,
        freq_mask_param,
        time_mask_param,
        n_freq_masks=1,
        n_time_masks=1,
        mask_value=0.0,
        data_format='default',
        **kwargs,
    ):

        backend.validate_data_format_str(data_format)

        super(SpecAugment, self).__init__(**kwargs)

        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.mask_value = mask_value

        if not self.freq_mask_param or not self.time_mask_param:
            raise RuntimeError(
                "Both freq_mask_param and time_mask_param must be defined and different "
                "than zero"
            )

        self.data_format = K.image_data_format() if data_format == _CH_DEFAULT_STR else data_format

    @staticmethod
    def _generate_axis_mask(inputs):
        """
        Generate a mask for the axis provided
        Args:
            inputs (`tuple`): A 3-tuple with the following structure:
                inputs[0] (float `Tensor`): A spectrogram. Its shape is (time, freq, ch) or (ch, time, freq)
                    depending on data_format
                inputs[1] (int): The axis limit. If mask will be applied to time axis it will be `time`, if it will
                    be applied to frequency axis, then it will be `freq`
                inputs[2] (int `Tensor`): The axis indices. We need this Tensor of indices to indicate where to apply
                    the mask.
                inputs[3] (int): The mask param as defined in the original paper, which is the max width of the mask
                    applied.
        Returns:
            (bool `Tensor`): A boolean tensor representing the mask. Its shape is (time, freq, ch) or (ch, time, freq)
                depending on inputs[0] shape (that is, the input spectrogram).
        """
        x, axis_limit, axis_indices, mask_param = inputs

        mask_width = tf.random.uniform(shape=(), maxval=mask_param, dtype=tf.int32)
        mask_start = tf.random.uniform(shape=(), maxval=axis_limit - mask_width, dtype=tf.int32)

        return tf.logical_and(axis_indices >= mask_start, axis_indices <= mask_start + mask_width)

    def _apply_masks_to_axis(self, x, axis, mask_param, n_masks):
        """
        Applies a number of masks (defined by the parameter n_masks) to the spectrogram
        by the axis provided.
        Args:
            x (float `Tensor`): A spectrogram. Its shape is (time, freq, ch) or (ch, time, freq)
                    depending on data_format.
            axis (int): The axis where the masks will be applied
            mask_param (int): The mask param as defined in the original paper, which is the max width of the mask
                    applied to the specified axis.
            n_masks (int): The number of masks to be applied

        Returns:
            (float `Tensor`): The masked spectrogram. Its shape is (time, freq, ch) or (ch, time, freq)
                depending on x shape (that is, the input spectrogram).
        """
        axis_limit = K.int_shape(x)[axis]
        axis_indices = tf.range(axis_limit)

        if axis == 0:
            axis_indices = tf.reshape(axis_indices, (-1, 1, 1))
        elif axis == 1:
            axis_indices = tf.reshape(axis_indices, (1, -1, 1))
        elif axis == 2:
            axis_indices = tf.reshape(axis_indices, (1, 1, -1))
        else:
            raise NotImplementedError(f"Axis parameter must be one of the following: 0, 1, 2")

        # Check if mask_width is greater than axis_limit
        if axis_limit < mask_param:
            raise ValueError(
                "Time and freq axis shapes must be greater than time_mask_param "
                "and freq_mask_param respectively"
            )

        x_repeated = tf.repeat(tf.expand_dims(x, 0), n_masks, axis=0)
        axis_limit_repeated = tf.repeat(axis_limit, n_masks, axis=0)
        axis_indices_repeated = tf.repeat(tf.expand_dims(axis_indices, 0), n_masks, axis=0)
        mask_param_repeated = tf.repeat(mask_param, n_masks, axis=0)

        masks = tf.map_fn(
            elems=(x_repeated, axis_limit_repeated, axis_indices_repeated, mask_param_repeated),
            fn=self._generate_axis_mask,
            dtype=(tf.float32, tf.int32, tf.int32, tf.int32),
            fn_output_signature=tf.bool,
        )

        mask = tf.math.reduce_any(masks, 0)
        return tf.where(mask, self.mask_value, x)

    def _apply_spec_augment(self, x):
        """
        Main method that applies SpecAugment technique by both frequency and
        time axis.
        Args:
            x (float `Tensor`) : A spectrogram. Its shape is (time, freq, ch) or (ch, time, freq)
                    depending on data_format.
        Returns:
            (float `Tensor`): The spectrogram masked by time and frequency axis. Its shape is (time, freq, ch)
                or (ch, time, freq) depending on x shape (that is, the input spectrogram).
        """
        if self.data_format == _CH_LAST_STR:
            time_axis, freq_axis = 0, 1
        else:
            time_axis, freq_axis = 1, 2

        if self.n_time_masks >= 1:
            x = self._apply_masks_to_axis(
                x, axis=time_axis, mask_param=self.time_mask_param, n_masks=self.n_time_masks
            )
        if self.n_freq_masks >= 1:
            x = self._apply_masks_to_axis(
                x, axis=freq_axis, mask_param=self.freq_mask_param, n_masks=self.n_freq_masks
            )
        return x

    def call(self, x, training=None, **kwargs):
        if training in (None, False):
            return x

        if K.ndim(x) != 4:
            raise ValueError(
                'ndim of input tensor x should be 4 (batch spectrogram),' 'but it is %d' % K.ndim(x)
            )

        ch_axis = 1 if self.data_format == 'channels_first' else 3

        if K.int_shape(x)[ch_axis] != 1:
            raise RuntimeError(
                'SpecAugment does not support spectrograms with depth greater than 1'
            )

        return tf.map_fn(
            elems=x, fn=self._apply_spec_augment, dtype=tf.float32, fn_output_signature=tf.float32
        )

    def get_config(self):
        config = super(SpecAugment, self).get_config()
        config.update(
            {
                'freq_mask_param': self.freq_mask_param,
                'time_mask_param': self.time_mask_param,
                'n_freq_masks': self.n_freq_masks,
                'n_time_masks': self.n_time_masks,
                'mask_value': self.mask_value,
                'data_format': self.data_format,
            }
        )
        return config
