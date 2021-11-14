"""Time-frequency Keras layers.

This module has low-level implementations of some popular time-frequency operations such as STFT and inverse STFT.
We're using these layers to compose layers in `kapre.composed` where more high-level and popular layers
such as melspectrogram layer are provided. You should go check it out!

Note:
    **Why time-frequency representation?**

    Every representation (STFT, melspectrogram, etc) has something in common - they're all 2D representations
    (time, frequency-ish) of audio signals. They're helpful because they decompose an audio signal, which is a simultaneous
    mixture of a lot of frequency components into different frequency bins. They have spatial property; the frequency
    bins are *sorted*, so frequency bins nearby has represent only slightly different frequency components. The
    frequency decomposition is also what's happening during human auditory perception through cochlea.

    **Which representation to use as input?**

    For a quick summary, check out my tutorial paper, `A Tutorial on Deep Learning for Music Information Retrieval <https://arxiv.org/abs/1709.04396>`_.

"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

from . import backend
from .backend import _CH_FIRST_STR, _CH_LAST_STR, _CH_DEFAULT_STR
from .tflite_compatible_stft import atan2_tflite

__all__ = [
    'STFT',
    'InverseSTFT',
    'Magnitude',
    'Phase',
    'MagnitudeToDecibel',
    'ApplyFilterbank',
    'Delta',
    'ConcatenateFrequencyMap',
]


def _shape_spectrum_output(spectrums, data_format):
    """Shape batch spectrograms into the right format.

    Args:
        spectrums (`Tensor`): result of tf.signal.stft or similar, i.e., (..., time, freq).
        data_format (`str`): 'channels_first' or 'channels_last'

    Returns:
        spectrums (`Tensor`): a transposed version of input `spectrums`

    """
    if data_format == _CH_FIRST_STR:
        pass  # probably it's already (batch, channel, time, freq)
    else:
        spectrums = tf.transpose(spectrums, perm=(0, 2, 3, 1))  # (batch, time, freq, channel)
    return spectrums


class STFT(Layer):
    """
    A Short-time Fourier transform layer.

    It uses `tf.signal.stft` to compute complex STFT. Additionally, it reshapes the output to be a proper 2D batch.

    If `output_data_format == 'channels_last'`, the output shape is (batch, time, freq, channel)
    If `output_data_format == 'channels_first'`, the output shape is (batch, channel, time, freq)

    Args:
        n_fft (int): Number of FFTs. Defaults to `2048`
        win_length (int or None): Window length in sample. Defaults to `n_fft`.
        hop_length (int or None): Hop length in sample between analysis windows. Defaults to `win_length // 4` following Librosa.
        window_name (str or None): *Name* of `tf.signal` function that returns a 1D tensor window that is used in analysis.
            Defaults to `hann_window` which uses `tf.signal.hann_window`.
            Window availability depends on Tensorflow version. More details are at `kapre.backend.get_window()`.
        pad_begin (bool): Whether to pad with zeros along time axis (length: win_length - hop_length). Defaults to `False`.
        pad_end (bool): Whether to pad with zeros at the finishing end of the signal.
        input_data_format (str): the audio data format of input waveform batch.
            `'channels_last'` if it's `(batch, time, channels)` and
            `'channels_first'` if it's `(batch, channels, time)`.
            Defaults to the setting of your Keras configuration. (`tf.keras.backend.image_data_format()`)
        output_data_format (str): The data format of output STFT.
            `'channels_last'` if you want `(batch, time, frequency, channels)` and
            `'channels_first'` if you want `(batch, channels, time, frequency)`
            Defaults to the setting of your Keras configuration. (`tf.keras.backend.image_data_format()`)

        **kwargs: Keyword args for the parent keras layer (e.g., `name`)

    Example:
        ::

            input_shape = (2048, 1)  # mono signal
            model = Sequential()
            model.add(kapre.STFT(n_fft=1024, hop_length=512, input_shape=input_shape))
            # now the shape is (batch, n_frame=3, n_freq=513, ch=1)
            # and the dtype is complex

    """

    def __init__(
        self,
        n_fft=2048,
        win_length=None,
        hop_length=None,
        window_name=None,
        pad_begin=False,
        pad_end=False,
        input_data_format='default',
        output_data_format='default',
        **kwargs,
    ):
        super(STFT, self).__init__(**kwargs)

        backend.validate_data_format_str(input_data_format)
        backend.validate_data_format_str(output_data_format)

        if win_length is None:
            win_length = n_fft
        if hop_length is None:
            hop_length = win_length // 4

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window_name = window_name
        self.window_fn = backend.get_window_fn(window_name)
        self.pad_begin = pad_begin
        self.pad_end = pad_end

        idt, odt = input_data_format, output_data_format
        self.output_data_format = K.image_data_format() if odt == _CH_DEFAULT_STR else odt
        self.input_data_format = K.image_data_format() if idt == _CH_DEFAULT_STR else idt

    def call(self, x):
        """
        Compute STFT of the input signal. If the `time` axis is not the last axis of `x`, it should be transposed first.

        Args:
            x (float `Tensor`): batch of audio signals, (batch, ch, time) or (batch, time, ch) based on input_data_format

        Return:
            (complex `Tensor`): A STFT representation of x in a 2D batch shape.
            `complex64` if `x` is `float32`, `complex128` if `x` is `float64`.
            Its shape is (batch, time, freq, ch) or (batch. ch, time, freq) depending on `output_data_format` and
            `time` is the number of frames, which is `((len_src + (win_length - hop_length) / hop_length) // win_length )` if `pad_end` is `True`.
            `freq` is the number of fft unique bins, which is `n_fft // 2 + 1` (the unique components of the FFT).
        """
        waveforms = x  # (batch, ch, time) if input_data_format == 'channels_first'.
        # (batch, time, ch) if input_data_format == 'channels_last'.

        # this is needed because tf.signal.stft lives in channels_first land.
        if self.input_data_format == _CH_LAST_STR:
            waveforms = tf.transpose(
                waveforms, perm=(0, 2, 1)
            )  # always (batch, ch, time) from here

        if self.pad_begin:
            waveforms = tf.pad(
                waveforms, tf.constant([[0, 0], [0, 0], [int(self.n_fft - self.hop_length), 0]])
            )

        stfts = tf.signal.stft(
            signals=waveforms,
            frame_length=self.win_length,
            frame_step=self.hop_length,
            fft_length=self.n_fft,
            window_fn=self.window_fn,
            pad_end=self.pad_end,
            name='%s_tf.signal.stft' % self.name,
        )  # (batch, ch, time, freq)

        if self.output_data_format == _CH_LAST_STR:
            stfts = tf.transpose(stfts, perm=(0, 2, 3, 1))  # (batch, t, f, ch)

        return stfts

    def get_config(self):
        config = super(STFT, self).get_config()
        config.update(
            {
                'n_fft': self.n_fft,
                'win_length': self.win_length,
                'hop_length': self.hop_length,
                'window_name': self.window_name,
                'pad_begin': self.pad_begin,
                'pad_end': self.pad_end,
                'input_data_format': self.input_data_format,
                'output_data_format': self.output_data_format,
            }
        )
        return config


class InverseSTFT(Layer):
    """An inverse-STFT layer.

    If `output_data_format == 'channels_last'`, the output shape is (batch, time, channel)
    If `output_data_format == 'channels_first'`, the output shape is (batch, channel, time)

    Note that the result of inverse STFT could be longer than the original signal due to the padding. Do check the
    size of the result by yourself and trim it if needed.

    Args:
        n_fft (int): Number of FFTs. Defaults to `2048`
        win_length (`int` or `None`): Window length in sample. Defaults to `n_fft`.
        hop_length (`int` or `None`): Hop length in sample between analysis windows. Defaults to `n_fft // 4` following Librosa.
        forward_window_name (str or None): *Name* of `tf.signal` function that *was* used in the forward STFT.
            Defaults to `hann_window`, assuming `tf.signal.hann_window` was used.
            Window availability depends on Tensorflow version. More details are at `kapre.backend.get_window()`.

        input_data_format (`str`): the data format of input STFT batch
            `'channels_last'` if you want `(batch, time, frequency, channels)`
            `'channels_first'` if you want `(batch, channels, time, frequency)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())
        output_data_format (`str`): the audio data format of output waveform batch.
            `'channels_last'` if it's `(batch, time, channels)`
            `'channels_first'` if it's `(batch, channels, time)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())

        **kwargs: Keyword args for the parent keras layer (e.g., `name`)

    Example:
        ::

            input_shape = (3, 513, 1)  # 3 frames, 513 frequency bins, 1 channel
            # and input dtype is complex
            model = Sequential()
            model.add(kapre.InverseSTFT(n_fft=1024, hop_length=512, input_shape=input_shape))
            # now the shape is (batch, time=2048, ch=1)

    """

    def __init__(
        self,
        n_fft=2048,
        win_length=None,
        hop_length=None,
        forward_window_name=None,
        input_data_format='default',
        output_data_format='default',
        **kwargs,
    ):
        super(InverseSTFT, self).__init__(**kwargs)

        backend.validate_data_format_str(input_data_format)
        backend.validate_data_format_str(output_data_format)

        if win_length is None:
            win_length = n_fft
        if hop_length is None:
            hop_length = win_length // 4

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.forward_window_name = forward_window_name
        self.window_fn = tf.signal.inverse_stft_window_fn(
            frame_step=hop_length, forward_window_fn=backend.get_window_fn(forward_window_name)
        )

        idt, odt = input_data_format, output_data_format
        self.output_data_format = K.image_data_format() if odt == _CH_DEFAULT_STR else odt
        self.input_data_format = K.image_data_format() if idt == _CH_DEFAULT_STR else idt

    def call(self, x):
        """
        Compute inverse STFT of the input STFT.

        Args:
            x (complex `Tensor`): batch of STFTs, (batch, ch, time, freq) or (batch, time, freq, ch) depending on `input_data_format`

        Return:
            (`float`): audio signals of x. Shape: 1D batch shape. I.e., (batch, time, ch) or (batch, ch, time) depending on `output_data_format`

        """
        stfts = x  # (batch, ch, time, freq) if input_data_format == 'channels_first'.
        # (batch, time, freq, ch) if input_data_format == 'channels_last'.

        # this is needed because tf.signal.stft lives in channels_first land.
        if self.input_data_format == _CH_LAST_STR:
            stfts = tf.transpose(stfts, perm=(0, 3, 1, 2))  # now always (b, ch, t, f)

        waveforms = tf.signal.inverse_stft(
            stfts=stfts,
            frame_length=self.win_length,
            frame_step=self.hop_length,
            fft_length=self.n_fft,
            window_fn=self.window_fn,
            name='%s_tf.signal.istft' % self.name,
        )  # (batch, ch, time)

        if self.output_data_format == _CH_LAST_STR:
            waveforms = tf.transpose(waveforms, perm=(0, 2, 1))  # (batch, time, ch)

        return waveforms

    def get_config(self):
        config = super(InverseSTFT, self).get_config()
        config.update(
            {
                'n_fft': self.n_fft,
                'win_length': self.win_length,
                'hop_length': self.hop_length,
                'forward_window_name': self.forward_window_name,
                'input_data_format': self.input_data_format,
                'output_data_format': self.output_data_format,
            }
        )
        return config


class Magnitude(Layer):
    """Compute the magnitude of the complex input, resulting in a float tensor

    Example:
        ::

            input_shape = (2048, 1)  # mono signal
            model = Sequential()
            model.add(kapre.STFT(n_fft=1024, hop_length=512, input_shape=input_shape))
            mode.add(Magnitude())
            # now the shape is (batch, n_frame=3, n_freq=513, ch=1) and dtype is float

    """

    def call(self, x):
        """
        Args:
            x (complex `Tensor`): input complex tensor

        Returns:
            (float `Tensor`): magnitude of `x`
        """
        return tf.abs(x)


class Phase(Layer):
    """Compute the phase of the complex input in radian, resulting in a float tensor

    Includes option to use approximate phase algorithm this will return the same
    results as the PhaseTflite layer (the tflite compatible layer).

    Args:
        approx_atan_accuracy (`int`): if `None` will use tf.math.angle() to
            calculate the phase accurately. If an `int` this is the number of
            iterations to calculate the approximate atan() using a tflite compatible
            method. the higher the number the more accurate e.g.
            approx_atan_accuracy=29000. You may want to experiment with adjusting
            this number: trading off accuracy with inference speed.

    Example:
        ::

            input_shape = (2048, 1)  # mono signal
            model = Sequential()
            model.add(kapre.STFT(n_fft=1024, hop_length=512, input_shape=input_shape))
            model.add(Phase())
            # now the shape is (batch, n_frame=3, n_freq=513, ch=1) and dtype is float
    """

    def __init__(self, approx_atan_accuracy=None, **kwargs):
        super(Phase, self).__init__(**kwargs)
        self.approx_atan_accuracy = approx_atan_accuracy

    def call(self, x):
        """
        Args:
            x (complex `Tensor`): input complex tensor

        Returns:
            (float `Tensor`): phase of `x` (Radian)
        """
        if self.approx_atan_accuracy:
            return atan2_tflite(tf.math.imag(x), tf.math.real(x), n=self.approx_atan_accuracy)

        return tf.math.angle(x)

    def get_config(self):
        config = super(Phase, self).get_config()
        config.update(
            {
                'tflite_phase_accuracy': self.approx_atan_accuracy,
            }
        )
        return config


class MagnitudeToDecibel(Layer):
    """A class that wraps `backend.magnitude_to_decibel` to compute decibel of the input magnitude.

    Args:
        ref_value (`float`): an input value that would become 0 dB in the result.
            For spectrogram magnitudes, ref_value=1.0 usually make the decibel-scaled output to be around zero
            if the input audio was in [-1, 1].
        amin (`float`): the noise floor of the input. An input that is smaller than `amin`, it's converted to `amin.
        dynamic_range (`float`): range of the resulting value. E.g., if the maximum magnitude is 30 dB,
            the noise floor of the output would become (30 - dynamic_range) dB

    Example:
        ::

            input_shape = (2048, 1)  # mono signal
            model = Sequential()
            model.add(kapre.STFT(n_fft=1024, hop_length=512, input_shape=input_shape))
            model.add(Magnitude())
            model.add(MagnitudeToDecibel())
            # now the shape is (batch, n_frame=3, n_freq=513, ch=1) and dtype is float

    """

    def __init__(self, ref_value=1.0, amin=1e-5, dynamic_range=80.0, **kwargs):
        super(MagnitudeToDecibel, self).__init__(**kwargs)
        self.ref_value = ref_value
        self.amin = amin
        self.dynamic_range = dynamic_range

    def call(self, x):
        """
        Args:
            x (`Tensor`): float tensor. Can be batch or not. Something like magnitude of STFT.

        Returns:
            (`Tensor`): decibel-scaled float tensor of `x`.
        """
        return backend.magnitude_to_decibel(
            x, ref_value=self.ref_value, amin=self.amin, dynamic_range=self.dynamic_range
        )

    def get_config(self):
        config = super(MagnitudeToDecibel, self).get_config()
        config.update(
            {
                'amin': self.amin,
                'dynamic_range': self.dynamic_range,
                'ref_value': self.ref_value,
            }
        )
        return config


class ApplyFilterbank(Layer):
    """
    Apply a filterbank to the input spectrograms.

    Args:
        filterbank (`Tensor`): filterbank tensor in a shape of (n_freq, n_filterbanks)
        data_format (`str`): specifies the data format of batch input/output
        **kwargs: Keyword args for the parent keras layer (e.g., `name`)

    Example:
        ::

            input_shape = (2048, 1)  # mono signal
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
            model.add(ApplyFilterbank(type='mel', filterbank_kwargs=kwargs))
            # (batch, n_frame=3, n_mels=128, ch=1)


    """

    def __init__(
        self,
        type,
        filterbank_kwargs,
        data_format='default',
        **kwargs,
    ):

        backend.validate_data_format_str(data_format)

        self.type = type
        self.filterbank_kwargs = filterbank_kwargs

        if type == 'log':
            self.filterbank = _log_filterbank = backend.filterbank_log(**filterbank_kwargs)
        elif type == 'mel':
            self.filterbank = _mel_filterbank = backend.filterbank_mel(**filterbank_kwargs)

        if data_format == _CH_DEFAULT_STR:
            self.data_format = K.image_data_format()
        else:
            self.data_format = data_format

        if self.data_format == _CH_FIRST_STR:
            self.freq_axis = 3
        else:
            self.freq_axis = 2
        super(ApplyFilterbank, self).__init__(**kwargs)

    def call(self, x):
        """
        Apply filterbank to `x`.

        Args:
            x (`Tensor`): float tensor in 2D batch shape.
        """

        # x: 2d batch input. (b, t, fr, ch) or (b, ch, t, fr)
        output = tf.tensordot(x, self.filterbank, axes=(self.freq_axis, 0))
        # ch_last -> (b, t, ch, new_fr). ch_first -> (b, ch, t, new_fr)
        if self.data_format == _CH_LAST_STR:
            output = tf.transpose(output, (0, 1, 3, 2))
        return output

    def get_config(self):
        config = super(ApplyFilterbank, self).get_config()
        config.update(
            {
                'type': self.type,
                'filterbank_kwargs': self.filterbank_kwargs,
                'data_format': self.data_format,
            }
        )
        return config


class Delta(Layer):
    """Calculates delta, a local estimate of the derivative along time axis.
    See torchaudio.functional.compute_deltas or librosa.feature.delta for more details.

    Args:
        win_length (int): Window length of the derivative estimation. Defaults to 5
        mode (`str`): Specifies pad mode of `tf.pad`. Case-insensitive. Defaults to 'symmetric'.
            Can be 'symmetric', 'reflect', 'constant', or whatever `tf.pad` supports.

    Example:
        ::

            input_shape = (2048, 1)  # mono signal
            model = Sequential()
            model.add(kapre.STFT(n_fft=1024, hop_length=512, input_shape=input_shape))
            model.add(kapre.Magnitude())
            model.add(Delta())
            # (batch, n_frame=3, n_freq=513, ch=1) and dtype is float

    """

    def __init__(self, win_length=5, mode='symmetric', data_format='default', **kwargs):
        backend.validate_data_format_str(data_format)

        if not win_length >= 3:
            raise ValueError(
                'win_length should be equal or bigger than 3, but it is %d' % win_length
            )
        if win_length % 2 != 1:
            raise ValueError('win_length should be an odd number, but it is %d' % win_length)
        if mode.lower() not in ('symmetric', 'reflect', 'constant'):
            raise ValueError(
                'mode.lower() should be one of {}'.format(str(('symmetric', 'reflect', 'constant')))
                + 'but it is {}'.format(mode)
            )

        if data_format == _CH_DEFAULT_STR:
            self.data_format = K.image_data_format()
        else:
            self.data_format = data_format

        self.win_length = win_length
        self.mode = mode
        self.n = (self.win_length - 1) // 2  # half window length
        self.denom = 2 * sum([_n ** 2 for _n in range(1, self.n + 1, 1)])  # denominator
        super(Delta, self).__init__(**kwargs)

    def call(self, x):
        """
        Args:
            x (`Tensor`): a 2d batch (b, t, f, ch) or (b, ch, t, f)

        Returns:
            (`Tensor`): A tensor with the same shape as input data.
        """
        if self.data_format == 'channels_first':
            x = K.permute_dimensions(x, (0, 2, 3, 1))

        x = tf.pad(
            x, tf.constant([[0, 0], [self.n, self.n], [0, 0], [0, 0]]), mode=self.mode
        )  # pad over time
        kernel = K.arange(-self.n, self.n + 1, 1, dtype=K.floatx())
        kernel = K.reshape(kernel, (-1, 1, 1, 1))  # time, freq, in_ch, out_ch

        x = K.conv2d(x, kernel, data_format=_CH_LAST_STR) / self.denom
        if self.data_format == _CH_FIRST_STR:
            x = K.permute_dimensions(x, (0, 3, 1, 2))

        return x

    def get_config(self):
        config = super(Delta, self).get_config()
        config.update(
            {'win_length': self.win_length, 'mode': self.mode, 'data_format': self.data_format}
        )

        return config


class ConcatenateFrequencyMap(Layer):
    """Addes a frequency information channel to spectrograms.

    The added frequency channel (=frequency map) has a linearly increasing values from 0.0 to 1.0,
    indicating the normalize frequency of a time-frequency bin. This layer can be applied to input audio spectrograms
    or any feature maps so that the following layers can be conditioned on the frequency. (Imagine something like
    positional encoding in NLP but the position is on frequency axis).

    A combination of `ConcatenateFrequencyMap` and `Conv2D` is known as frequency-aware convolution (see References).
    For your convenience, such a layer is supported by `karep.composed.get_frequency_aware_conv2d()`.

    Args:
        data_format (str): specifies the data format of batch input/output.
        **kwargs: Keyword args for the parent keras layer (e.g., `name`)

    Example:
        ::

            input_shape = (2048, 1)  # mono signal
            model = Sequential()
            model.add(kapre.STFT(n_fft=1024, hop_length=512, input_shape=input_shape))
            model.add(kapre.Magnitude())
            # (batch, n_frame=3, n_freq=513, ch=1) and dtype is float
            model.add(kapre.ConcatenateFrequencyMap())
            # (batch, n_frame=3, n_freq=513, ch=2)
            # now add your model
            mode.add(keras.layers.Conv2D(16, (3, 3), strides=(2, 2), activation='relu')
            # you can concatenate frequency map before other conv layers,
            # but probably, you wouldn't want to add it right before batch normalization.
            model.add(kapre.ConcatenateFrequencyMap())
            model.add(keras.layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu')
            model.add(keras.layers.MaxPooling2D((2, 2)))  # length of frequency axis doesn't matter

    References:
        Koutini, K., Eghbal-zadeh, H., & Widmer, G. (2019).
        `Receptive-Field-Regularized CNN Variants for Acoustic Scene Classification <https://arxiv.org/abs/1909.02859>`_.
        In Proceedings of the Detection and Classification of Acoustic Scenes and Events 2019 Workshop (DCASE2019).

    """

    def __init__(self, data_format='default', **kwargs):
        backend.validate_data_format_str(data_format)

        if data_format == _CH_DEFAULT_STR:
            self.data_format = K.image_data_format()
        else:
            self.data_format = data_format

        self.data_format = data_format

        super(ConcatenateFrequencyMap, self).__init__(**kwargs)

    def call(self, x):
        """
        Args:
            x (`Tensor`): a 2d batch (b, t, f, ch) or (b, ch, t, f)

        Returns:
            x (`Tensor`): a 2d batch (b, t, f, ch + 1) or (b, ch + 1, t, f)
        """
        return self._concat_frequency_map(x)

    def _concat_frequency_map(self, inputs):
        shape = tf.shape(inputs)
        time_axis, freq_axis, ch_axis = (1, 2, 3) if self.data_format == _CH_LAST_STR else (2, 3, 1)
        batch_size, n_freq, n_time, n_ch = (
            shape[0],
            shape[freq_axis],
            shape[time_axis],
            shape[ch_axis],
        )

        # freq_info shape: n_freq
        freq_map_1d = tf.cast(tf.linspace(start=0.0, stop=1.0, num=n_freq), dtype=tf.float32)

        new_shape = (1, 1, -1, 1) if self.data_format == _CH_LAST_STR else (1, 1, 1, -1)
        freq_map_1d = tf.reshape(freq_map_1d, new_shape)  # 4D now

        multiples = (
            (batch_size, n_time, 1, 1)
            if self.data_format == _CH_LAST_STR
            else (batch_size, 1, n_time, 1)
        )
        freq_map_4d = tf.tile(freq_map_1d, multiples)

        return tf.concat([inputs, freq_map_4d], axis=ch_axis)

    def get_config(self):
        config = super(ConcatenateFrequencyMap, self).get_config()
        config.update(
            {
                'data_format': self.data_format,
            }
        )
        return config
