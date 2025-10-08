"""Signal layers.

This module includes Kapre layers that deal with audio signals (waveforms).

"""
from __future__ import annotations

from typing import Optional, Tuple, Any, Union
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

from . import backend
from .backend import _CH_FIRST_STR, _CH_LAST_STR, _CH_DEFAULT_STR


__all__ = ['Frame', 'Energy', 'MuLawEncoding', 'MuLawDecoding', 'LogmelToMFCC']


class Frame(Layer):
    """
    Frame input audio signal. It is a wrapper of `tf.signal.frame`.

    Args:
        frame_length (int): length of a frame
        hop_length (int): hop length aka frame rate
        pad_end (bool): whether to pad at the end of the signal of there would be a otherwise-discarded partial frame
        pad_value (int or float): value to use in the padding
        data_format (str): `channels_first`, `channels_last`, or `default`
        **kwargs: optional keyword args for `tf.keras.layers.Layer()`

    Example:
        ::

            input_shape = (2048, 1)  # mono signal
            model = Sequential()
            model.add(kapre.Frame(frame_length=1024, hop_length=512, input_shape=input_shape))
            # now the shape is (batch, n_frame=3, frame_length=1024, ch=1)

    """

    def __init__(
        self,
        frame_length: int,
        hop_length: int,
        pad_end: bool = False,
        pad_value: Union[int, float] = 0,
        data_format: str = 'default',
        **kwargs: Any,
    ) -> None:
        super(Frame, self).__init__(**kwargs)

        backend.validate_data_format_str(data_format)

        # Input validation
        if frame_length <= 0:
            raise ValueError(f'frame_length must be positive, got: {frame_length}')
        if hop_length <= 0:
            raise ValueError(f'hop_length must be positive, got: {hop_length}')
        if frame_length < hop_length:
            raise ValueError(
                f'frame_length ({frame_length}) must be >= hop_length ({hop_length})'
            )

        self.frame_length = frame_length
        self.hop_length = hop_length
        self.pad_end = pad_end
        self.pad_value = pad_value

        if data_format == _CH_DEFAULT_STR:
            self.data_format = backend._get_image_data_format()
        else:
            self.data_format = data_format

        if data_format == _CH_FIRST_STR:
            self.time_axis = 2  # batch, ch, time
        else:
            self.time_axis = 1  # batch, time, ch

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Compute framed audio signal.

        Args:
            x: Batch audio signal in the specified 1D format.

        Returns:
            A framed tensor. The shape is (batch, time (frames), frame_length, channel) if `channels_last`,
            or (batch, channel, time (frames), frame_length) if `channels_first`.
        """
        return tf.signal.frame(
            x,
            frame_length=self.frame_length,
            frame_step=self.hop_length,
            pad_end=self.pad_end,
            pad_value=self.pad_value,
            axis=self.time_axis,
        )

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Layer configuration dictionary.
        """
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
        data_format (str): `channels_first`, `channels_last`, or `default`
        **kwargs: optional keyword args for `tf.keras.layers.Layer()`

    Example:
        ::

            input_shape = (2048, 1)  # mono signal
            model = Sequential()
            model.add(kapre.Energy(frame_length=1024, hop_length=512, input_shape=input_shape))
            # now the shape is (batch, n_frame=3, ch=1)

    """

    def __init__(
        self,
        sample_rate: int = 22050,
        ref_duration: float = 0.1,
        frame_length: int = 2205,
        hop_length: int = 1102,
        pad_end: bool = False,
        pad_value: Union[int, float] = 0,
        data_format: str = 'default',
        **kwargs: Any,
    ) -> None:
        super(Energy, self).__init__(**kwargs)

        backend.validate_data_format_str(data_format)

        self.sample_rate = sample_rate
        self.ref_duration = ref_duration
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.pad_end = pad_end
        self.pad_value = pad_value

        if data_format == _CH_DEFAULT_STR:
            self.data_format = backend._get_image_data_format()
        else:
            self.data_format = data_format

        if data_format == _CH_FIRST_STR:
            self.time_axis = 2  # batch, ch, time
        else:
            self.time_axis = 1  # batch, time, ch

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Compute energy of each frame.

        Args:
            x: Batch audio signal in the specified 1D format.

        Returns:
            A tensor with frame energies. The shape is (batch, time (frames), channel) if `channels_last`, or
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

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Layer configuration dictionary.
        """
        config = super(Energy, self).get_config()
        config.update(
            {
                'sample_rate': self.sample_rate,
                'ref_duration': self.ref_duration,
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
        **kwargs: optional keyword args for `tf.keras.layers.Layer()`

    Note:
        Mu-law encoding was originally developed to increase signal-to-noise ratio of signal during transmission.
        In deep learning, mu-law became popular by `WaveNet <https://arxiv.org/abs/1609.03499>`_ where
        8-bit (256 channels) mu-law quantization was applied to the signal so that the generation of waveform amplitudes
        became a single-label 256-class classification problem.

    Example:
        ::

            input_shape = (2048, 1)  # mono signal (float in [-1, 1])
            model = Sequential()
            model.add(kapre.MuLawEncoding(quantization_channels=256, input_shape=input_shape))
            # now the shape is (batch, time=2048, ch=1) with int in [0, quantization_channels - 1]


    """

    def __init__(
        self,
        quantization_channels: int,
        **kwargs: Any,
    ) -> None:
        super(MuLawEncoding, self).__init__(**kwargs)

        # Input validation
        if quantization_channels < 2:
            raise ValueError(
                f'quantization_channels must be at least 2, got: {quantization_channels}'
            )
        if quantization_channels > 65536:  # Reasonable upper bound
            raise ValueError(
                f'quantization_channels must be <= 65536, got: {quantization_channels}'
            )

        self.quantization_channels = quantization_channels

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Encode signal using mu-law companding.

        Args:
            x: Audio signal to encode. Shape doesn't matter.

        Returns:
            Mu-law encoded signal. Shape doesn't change.
        """
        return backend.mu_law_encoding(x, self.quantization_channels)

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Layer configuration dictionary.
        """
        config = super(MuLawEncoding, self).get_config()
        config.update(
            {
                'quantization_channels': self.quantization_channels,
            }
        )

        return config


class MuLawDecoding(Layer):
    """
    Mu-law decoding (expansion) of mu-law encoded audio signal to [-1, 1].
    See `Wikipedia <https://en.wikipedia.org/wiki/Μ-law_algorithm>`_ for more details.

    Args:
        quantization_channels (positive int): Number of channels. For 8-bit encoding, use 256.
        **kwargs: optional keyword args for `tf.keras.layers.Layer()`

    Example:
        ::

            input_shape = (2048, 1)  # mono signal (int in [0, quantization_channels - 1])
            model = Sequential()
            model.add(kapre.MuLawDecoding(quantization_channels=256, input_shape=input_shape))
            # now the shape is (batch, time=2048, ch=1) with float dtype in [-1, 1]

    """

    def __init__(
        self,
        quantization_channels: int,
        **kwargs: Any,
    ) -> None:
        super(MuLawDecoding, self).__init__(**kwargs)
        self.quantization_channels = quantization_channels

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Decode mu-law encoded signal.

        Args:
            x: Mu-law encoded signal to decode. Shape doesn't matter.

        Returns:
            Decoded audio signal. Shape doesn't change.
        """
        return backend.mu_law_decoding(x, self.quantization_channels)

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Layer configuration dictionary.
        """
        config = super(MuLawDecoding, self).get_config()
        config.update(
            {
                'quantization_channels': self.quantization_channels,
            }
        )

        return config


class LogmelToMFCC(Layer):
    """
    Compute MFCC from log-melspectrogram.

    It wraps `tf.signal.mfccs_from_log_mel_spectrogram()`, which performs DCT-II.

    Note:
        In librosa, the DCT-II scales by `sqrt(1/n)` where `n` is the bin index of MFCC as it uses
        scipy. This is the correct orthogonal DCT.
        In Tensorflow though, because it follows HTK, it scales by `(0.5 * sqrt(2/n))`. This results in
        `sqrt(2)` scale difference in the first MFCC bins (`n=1`).

        As long as all of your data in training / inference / deployment is consistent (i.e., do not
        mix librosa and kapre MFCC), it'll be fine!

    Args:
        n_mfccs (int): Number of MFCC
        data_format (str): `channels_first`, `channels_last`, or `default`
        **kwargs: optional keyword args for `tf.keras.layers.Layer()`

    Example:
        ::

            input_shape = (40, 128, 1)  # mono melspectrogram with 40 frames and n_mels=128
            model = Sequential()
            model.add(kapre.LogmelToMFCC(n_mfccs=20, input_shape=input_shape))
            # now the shape is (batch, time=40, n_mfccs=20, ch=1)

    """

    def __init__(
        self,
        n_mfccs: int = 20,
        data_format: str = 'default',
        **kwargs: Any,
    ) -> None:
        super(LogmelToMFCC, self).__init__(**kwargs)
        backend.validate_data_format_str(data_format)

        self.n_mfccs = n_mfccs
        self.permutation: Optional[Tuple[int, ...]] = None
        if data_format == _CH_DEFAULT_STR:
            self.data_format = backend._get_image_data_format()
        else:
            self.data_format = data_format

        if self.data_format == _CH_LAST_STR:
            self.permutation = (0, 1, 3, 2)
        else:
            self.permutation = None

    def call(self, log_melgrams: tf.Tensor) -> tf.Tensor:
        """Compute MFCCs from log-melspectrogram.

        Args:
            log_melgrams: A batch of log-melgrams. `(b, time, mel, ch)` if `channels_last`
                and `(b, ch, time, mel)` if `channels_first`.

        Returns:
            MFCCs. `(batch, time, n_mfccs, ch)` if `channels_last`, `(batch, ch, time, n_mfccs)` if `channels_first`.
        """
        if self.permutation is not None:  # reshape so that last channel == mel
            log_melgrams = K.permute_dimensions(log_melgrams, pattern=self.permutation)

        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_melgrams)
        mfccs = mfccs[..., : self.n_mfccs]

        if self.permutation is not None:
            mfccs = K.permute_dimensions(mfccs, pattern=self.permutation)

        return mfccs

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration.

        Returns:
            Layer configuration dictionary.
        """
        config = super(LogmelToMFCC, self).get_config()
        config.update({'n_mfccs': self.n_mfccs, 'data_format': self.data_format})

        return config
