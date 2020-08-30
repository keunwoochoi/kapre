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
import warnings
import tensorflow as tf
from tensorflow.keras.layers import Layer
from . import backend
from tensorflow.keras import backend as K
from .backend import _CH_FIRST_STR, _CH_LAST_STR, _CH_DEFAULT_STR


__all__ = [
    'STFT',
    'InverseSTFT',
    'Magnitude',
    'Phase',
    'MagnitudeToDecibel',
    'ApplyFilterbank',
    'Delta',
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
    A Shor-time Fourier transform layer.

    It uses `tf.signal.stft` to compute complex STFT. Additionally, it reshapes the output to be a proper 2D batch.

    If `output_data_format == 'channels_last'`, the output shape is (batch, time, freq, channel)
    If `output_data_format == 'channels_first'`, the output shape is (batch, channel, time, freq)

    Args:
        n_fft (int): Number of FFTs. Defaults to `2048`
        win_length (int or None): Window length in sample. Defaults to `n_fft`.
        hop_length (int or None): Hop length in sample between analysis windows. Defaults to `n_fft // 4` following Librosa.
        window_fn (function or None): A function that returns a 1D tensor window that is used in analysis. Defaults to `tf.signal.hann_window`
        pad_begin (bool): Whether to pad with zeros along time axis (legnth: win_length - hop_length). Defaults to `False`.
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
        window_fn=None,
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
        if window_fn is None:
            window_fn = tf.signal.hann_window

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window_fn = window_fn
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
            `complex64` if `x` is `float32`. `complex128` if `x` is `float64`.
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
                'window_fn': self.window_fn,
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
        window_fn (function or `None`): A function that returns a 1D tensor window. Defaults to `tf.signal.hann_window`, but
            this default setup does NOT lead to perfect reconstruction.
            For perfect reconstruction, this should be set using `tf.signal.inverse_stft_window_fn()` with the
            correct `frame_step` and `forward_window_fn` that are matched to those used during STFT.
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
        window_fn=None,
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
        if window_fn is None:
            window_fn = tf.signal.hann_window
            warnings.warn(
                'In InverseSTFT, forward_window_fn was not set, hence the default hann window function'
                'is used. This would lead to non-perfect reconstruction of a STFT-ISTFT chain.'
                'For perfect reconstruction, forward_window_fn should be set using'
                'tf.signal.inverse_stft_window_fn with frame_step and forward_window_fn correctly specified.'
                'For more details, see how kapre.composed.get_perfectly_reconstructing_stft_istft() is'
                'implemented. '
            )

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window_fn = window_fn

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
                'window_fn': self.window_fn,
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

    Example:
        ::

            input_shape = (2048, 1)  # mono signal
            model = Sequential()
            model.add(kapre.STFT(n_fft=1024, hop_length=512, input_shape=input_shape))
            model.add(Phase())
            # now the shape is (batch, n_frame=3, n_freq=513, ch=1) and dtype is float

    """

    def call(self, x):
        """
        Args:
            x (complex `Tensor`): input complex tensor

        Returns:
            (float `Tensor`): phase of `x` (Radian)
        """
        return tf.math.angle(x)


class MagnitudeToDecibel(Layer):
    """A class that wraps `backend.magnitude_to_decibel` to compute decibel of the input magnitude.

    Args:
        ref_value (`float`): an input value that would become 0 dB in the result.
            For spectrogram magnitudes, ref_value=1.0 usually make the decibel-sclaed output to be around zero
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
            {'amin': self.amin, 'dynamic_range': self.dynamic_range, 'ref_value': self.ref_value,}
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
        self, type, filterbank_kwargs, data_format='default', **kwargs,
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
