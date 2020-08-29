"""Signal layers.

This module includes Kapre layers that process audio signals (waveforms).

"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
from . import backend
from tensorflow.keras import backend as K
from .backend import _CH_FIRST_STR, _CH_LAST_STR, _CH_DEFAULT_STR


class Frame(Layer):
    """
    Frame input audio signal. It is a wrapper of `tf.signal.frame`.

    Args:
        frame_length (int): length of a frame
        hop_length (int): hop length aka frame rate
        pad_end (bool): whether to pad at the end of the signal of there would be a otherwise-discarded partial frame
        pad_value (int or float): value to use in the padding
        data_format (str): 'channels_first', 'channels_last', or `default`
            **kwargs:
    """

    def __init__(
        self, frame_length, hop_length, pad_end=False, pad_value=0, data_format='default', **kwargs
    ):
        super(Frame, self).__init__(**kwargs)

        backend.validate_data_format_str(data_format)

        self.frame_length = frame_length
        self.hop_length = hop_length
        self.pad_end = pad_end
        self.pad_value = pad_value

        if data_format == _CH_DEFAULT_STR:
            self.data_format = K.image_data_format()
        else:
            self.data_format = data_format

        if data_format == _CH_FIRST_STR:
            self.time_axis = 2  # batch, ch, time
        else:
            self.time_axis = 1  # batch, time, ch

    def call(self, x):
        """
        Args:
            x (`Tensor`): batch audio signal in the specified 1D format in initiation.

        Returns: (`Tensor`): A framed tensor. The shape is
            (batch, time (frames), frame_length, channel) if `channels_last`, and
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

    def get_config(self):
        config = super(Frame, self).get_config()
        config.update(
            {
                'frame_length': self.frame_length,
                'hop_length': self.hop_length,
                'pad_end': self.pad_end,
                'pad_value': self.pad_value,
                'data_format': self.data_format,
            }
        )

        return config


class Energy(Layer):
    """
    Compute energy of each frame. The energy computed for each frame then is normalized so that the values would
    represent energy per `ref_duration`. I.e., if `frame_length` > `sample_rate * ref_duration`,

    Args:
        sample_rate (int): sample rate of the audio
        ref_duration (float): reference duration for normalization
        frame_length (int): length of a frame that is used in computing energy
        hop_length (int): hop length aka frame rate. time resolution of the energy computation.
        pad_end (bool): whether to pad at the end of the signal of there would be a otherwise-discarded partial frame
        pad_value (int or float): value to use in the padding
        data_format (str): 'channels_first', 'channels_last', or `default`
        **kwargs:
    """

    def __init__(
        self,
        sample_rate=22050,
        ref_duration=0.1,
        frame_length=2205,
        hop_length=1102,
        pad_end=False,
        pad_value=0,
        data_format='default',
        **kwargs,
    ):
        super(Energy, self).__init__(**kwargs)

        backend.validate_data_format_str(data_format)

        self.sample_rate = sample_rate
        self.ref_duration = ref_duration
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.pad_end = pad_end
        self.pad_value = pad_value

        if data_format == _CH_DEFAULT_STR:
            self.data_format = K.image_data_format()
        else:
            self.data_format = data_format

        if data_format == _CH_FIRST_STR:
            self.time_axis = 2  # batch, ch, time
        else:
            self.time_axis = 1  # batch, time, ch

    def call(self, x):
        """
        Args:
            x (`Tensor`): batch audio signal in the specified 1D format in initiation.

        Returns: (`Tensor`): A framed tensor. The shape is
            (batch, time (frames), channel) if `channels_last`, and
            (batch, channel, time (frames)) if `channels_first`.
        """
        frames = tf.signal.frame(
            x,
            frame_length=self.frame_length,
            frame_step=self.hop_length,
            pad_end=self.pad_end,
            pad_value=self.pad_value,
            axis=self.time_axis,
        )
        frames = tf.math.square(frames)  # batch, ndim=4

        frame_axis = 2 if self.data_format == _CH_LAST_STR else 3
        energies = tf.math.reduce_sum(
            frames, axis=frame_axis
        )  # batch, ndim=3. (b, t, ch) or (b, ch, t)

        # normalize it to self.ref_duration
        nor_coeff = self.ref_duration / (self.frame_length / self.sample_rate)

        return nor_coeff * energies

    def get_config(self):
        config = super(Energy, self).get_config()
        config.update(
            {
                'frame_length': self.frame_length,
                'hop_length': self.hop_length,
                'pad_end': self.pad_end,
                'pad_value': self.pad_value,
                'data_format': self.data_format,
            }
        )

        return config


class MuLawEncoding(Layer):
    """
    Mu-law encoding (compression) of audio signal, in [-1, 1], to [0, quantization_channels - 1].
    See `Wikipedia <https://en.wikipedia.org/wiki/Μ-law_algorithm>`_ for more details.

    Args:
        quantization_channels (positive int): Number of channels. For 8-bit encoding, use 256.

    Note:
        Mu-law encoding was originally developed to increase signal-to-noise ratio of signal during transmission.
        In deep learning, mu-law became popular by `WaveNet <https://arxiv.org/abs/1609.03499>`_ where
        8-bit (256 channels) mu-law quantization was applied to the signal so that the generation of waveform amplitudes
        became a single-label 256-class classification problem.

    """

    def __init__(
        self, quantization_channels, **kwargs,
    ):
        super(MuLawEncoding, self).__init__(**kwargs)
        self.quantization_channels = quantization_channels

    def call(self, x):
        """

        Args:
            x (float `Tensor`): audio signal to encode. Shape doesn't matter.

        Returns:
            (int `Tensor`): mu-law encoded x. Shape doesn't change.
        """
        return backend.mu_law_encoding(x, self.quantization_channels)


class MuLawDecoding(Layer):
    """
    Mu-law decoding (expansion) of mu-law encoded audio signal to [-1, 1].
    See `Wikipedia <https://en.wikipedia.org/wiki/Μ-law_algorithm>`_ for more details.

    Args:
        quantization_channels (positive int): Number of channels. For 8-bit encoding, use 256.
    """

    def __init__(
        self, quantization_channels, **kwargs,
    ):
        super(MuLawDecoding, self).__init__(**kwargs)
        self.quantization_channels = quantization_channels

    def call(self, x):
        """

        Args:
            x (int `Tensor`): audio signal to decode. Shape doesn't matter.

        Returns:
            (float `Tensor`): mu-law encoded x. Shape doesn't change.
        """
        return backend.mu_law_decoding(x, self.quantization_channels)
