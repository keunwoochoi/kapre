import tensorflow as tf
from tensorflow.keras.layers import Layer
from . import backend
from tensorflow.keras import backend as K


def _shape_spectrum_output(spectrums, data_format):
    """spectrums: result of tf.signal.stft or similar, i.e., (..., time, freq)."""
    if data_format == 'channels_first':
        pass  # probably it's already (batch, channel, time, freq)
    else:
        spectrums = tf.transpose(spectrums, perm=(0, 2, 3, 1))  # (batch, time, freq, channel)
    return spectrums


class STFT(Layer):
    """
    A Shor-time Fourier transform layer.
    It uses `tf.signal.stft` to compute complex STFT. Additionally, it reshapes the output to be a proper 2D batch.
    If `channels_last`, the output shape is (batch, time, freq, channel)
    If `channels_first`, the output shape is (batch, channel, time, freq)

    Args:
        n_fft (int): Number of FFTs. Defaults to `2048`
        win_length (int or None): Window length in sample. Defaults to `n_fft`.
        hop_length (int or None): Hop length in sample between analysis windows. Defaults to `n_fft // 4` following Librosa.
        window_fn (function or None): A function that returns a 1D tensor window that is used in analysis. Defaults to `tf.signal.hann_window`
        pad_end (bool): Whether to pad with zeros at the finishing end of the signal.
        input_data_format (str): the audio data format of input waveform batch.
            `'channels_last'` if it's `(batch, time, channels)`
            `'channels_first'` if it's `(batch, channels, time)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())
        output_data_format (str): the data format of output mel spectrogram.
            `'channels_last'` if you want `(batch, time, frequency, channels)`
            `'channels_first'` if you want `(batch, channels, time, frequency)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())

        **kwargs: Keyword args for the parent keras layer (e.g., `name`)

    """

    # TODO: add pad_begin and pad zeros manually

    def __init__(
        self,
        n_fft=2048,
        win_length=None,
        hop_length=None,
        window_fn=None,
        pad_end=False,
        input_data_format='default',
        output_data_format='default',
        **kwargs,
    ):
        super(STFT, self).__init__(**kwargs)

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
        self.pad_end = pad_end

        idt, odt = input_data_format, output_data_format
        self.output_data_format = K.image_data_format() if odt == 'default' else odt
        self.input_data_format = K.image_data_format() if idt == 'default' else idt

    def call(self, x):
        """
        Compute STFT of the input signal. If the `time` axis is not the last axis of `x`, it should be transposed first.

        Args:
            x (float Tensor): batch of audio signals, (batch, ch, time) or (batch, time, ch) based on input_data_format

        Return:
            A STFT representation of x
                Shape: 2D batch shape. I.e., (batch, time, freq, ch) or (batch. ch, time, freq)
                Type: complex64/complex128 STFT values where fft_unique_bins is fft_length // 2 + 1
                (the unique components of the FFT).
        """
        signals = x  # (batch, ch, time) if input_data_format == 'channels_first'.
        # (batch, time, ch) if input_data_format == 'channels_last'.

        # this is needed because tf.signal.stft lives in channels_first land.
        if self.input_data_format == 'channels_last':
            signals = tf.transpose(signals, perm=(0, 2, 1))  # (batch, ch, time)

        stfts = tf.signal.stft(
            signals=signals,
            frame_length=self.win_length,
            frame_step=self.hop_length,
            fft_length=self.n_fft,
            window_fn=self.window_fn,
            pad_end=self.pad_end,
            name='%s_tf.signal.stft' % self.name,
        )  # (batch, ch, time, freq)

        if self.output_data_format == 'channels_last':
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
                'pad_end': self.pad_end,
                'input_data_format': self.input_data_format,
                'output_data_format': self.output_data_format,
            }
        )
        return config






class Magnitude(Layer):
    """Compute the magnitude of the complex input, resulting in a float tensor"""

    def call(self, x):
        return tf.abs(x)


class Phase(Layer):
    """Compute the phase of the complex input in radian, resulting in a float tensor"""

    def call(self, x):
        return tf.math.angle(x)


class MagnitudeToDecibel(Layer):
    """Wrap `backend.magnitude_to_decibel` to compute decibel of the input magnitude.
    It's basically 10 * log10(x) with some offset and noise floor.
    """

    def __init__(self, ref_value=1.0, amin=1e-5, dynamic_range=80.0, **kwargs):
        super(MagnitudeToDecibel, self).__init__(**kwargs)
        self.ref_value = ref_value
        self.amin = amin
        self.dynamic_range = dynamic_range

    def call(self, x):
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
        filterbank (tensor): filterbank tensor in a shape of (n_freq, n_filterbanks)
        data_format (str): specifies the data format of batch input/output
        **kwargs: Keyword args for the parent keras layer (e.g., `name`)

    """

    def __init__(
        self, type, filterbank_kwargs, data_format='default', **kwargs,
    ):
        self.type = type
        self.filterbank_kwargs = filterbank_kwargs

        if type == 'log':
            self.filterbank = _log_filterbank = backend.filterbank_log(**filterbank_kwargs)
        elif type == 'mel':
            self.filterbank = _mel_filterbank = backend.filterbank_mel(**filterbank_kwargs)

        if data_format == 'default':
            self.data_format = K.image_data_format()
        else:
            self.data_format = data_format

        if self.data_format == 'channels_first':
            self.freq_axis = 3
        else:
            self.freq_axis = 2
        super(ApplyFilterbank, self).__init__(**kwargs)

    def call(self, x):

        # x: 2d batch input. (b, t, fr, ch) or (b, ch, t, fr)
        output = tf.tensordot(x, self.filterbank, axes=(self.freq_axis, 0))
        # ch_last -> (b, t, ch, new_fr). ch_first -> (b, ch, t, new_fr)
        if self.data_format == 'channels_last':
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
        mode (str): Specifies pad mode of `tf.pad`. Case-insensitive. Defaults to 'symmetric'.
            Can be 'symmetric', 'reflect', 'constant', or whatever `tf.pad` supports.


    Returns:
        A tensor with the same shape as input data.

    """

    def __init__(self, win_length=5, mode='symmetric', data_format='default', **kwargs):

        assert data_format in ('default', 'channels_first', 'channels_last')
        assert win_length >= 3
        if win_length % 2 != 1:
            raise ValueError(
                'win_length is expected to be an odd number, but it is %d' % win_length
            )
        assert mode.lower() in ('symmetric', 'reflect', 'constant')

        if data_format == 'default':
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
            x (tensor): a 2d batch (b, t, f, ch) or (b, ch, t, f)

        """
        if self.data_format == 'channels_first':
            x = K.permute_dimensions(x, (0, 2, 3, 1))

        x = tf.pad(
            x, tf.constant([[0, 0], [self.n, self.n], [0, 0], [0, 0]]), mode=self.mode
        )  # pad over time
        kernel = K.arange(-self.n, self.n + 1, 1, dtype=K.floatx())
        kernel = K.reshape(kernel, (-1, 1, 1, 1))  # time, freq, in_ch, out_ch

        x = K.conv2d(x, kernel, data_format='channels_last') / self.denom
        if self.data_format == 'channels_first':
            x = K.permute_dimensions(x, (0, 3, 1, 2))

        return x

    def get_config(self):
        config = super(Delta, self).get_config()
        config.update(
            {'win_length': self.win_length, 'mode': self.mode, 'data_format': self.data_format}
        )

        return config
