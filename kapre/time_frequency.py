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
    It uses `tf.signal.stft` to compute complex STFT. Additionally, it reshapes the output to be a proper 2D batch. E.g., (batch, time, freq, channel) if `channels_last`.


    Args:
        n_fft (int): Number of FFTs. Defaults to `2048` following Librosa.
        win_length (int): Window length in sample. Defaults to `n_fft`.
        hop_length (int): Hop length in sample between analysis windows. Defaults to `n_fft // 4` following Librosa.
        window_fn: A function that returns a 1D tensor window that is used in analysis. Defaults to `tf.signal.hann_window`
        pad_end (bool): Whether to pad with zeros at the finishing end of the signal.

        **kwargs: Keyword args for keras layer (e.g., `name`)

    """

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
            x (float32/float64 Tensor): batch of audio signals, [..., samples].

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
            name='%s_tf.signal.stft' % self.name
        )  # (batch, ch, time, freq)

        if self.output_data_format == 'channels_last':
            stfts = tf.transpose(stfts, perm=(0, 2, 3, 1))  # (batch, t, f, ch)

        return stfts

    def get_config(self):
        config = {
            'n_fft': self.n_fft,
            'win_length': self.win_length,
            'hop_length': self.hop_length,
            'window_fn': self.window_fn,
            'pad_end': self.pad_end,
            'input_data_format': self.input_data_format,
            'output_data_format': self.output_data_format,
        }
        return config


class Magnitude(Layer):
    def call(self, x):
        return tf.abs(x)


class Phase(Layer):
    def call(self, x):
        return tf.math.angle(x)


class MagnitudeToDecibel(Layer):
    def __init__(self, amin=None, dynamic_range=120.0, **kwargs):
        super(MagnitudeToDecibel, self).__init__(**kwargs)
        self.amin = amin
        self.dynamic_range = dynamic_range

    def call(self, x):
        return backend.amplitude_to_decibel(x, amin=self.amin, dynamic_range=self.dynamic_range)

    def get_config(self):
        config = {
            'amin': self.amin,
            'dynamic_range': self.dynamic_range
        }
        return config


class ApplyFilterbank(Layer):
    """
    ### `Filterbank`

    `kapre.filterbank.Filterbank(n_fbs, trainable_fb, sr=None, init='mel', fmin=0., fmax=None,
                                 bins_per_octave=12, image_data_format='default', **kwargs)`

    #### Notes
        Input/output are 2D image format.
        E.g., if channel_first,
            - input_shape: ``(None, n_ch, time, n_freq)``
            - output_shape: ``(None, n_ch, time, n_new_bins)``


    #### Parameters
    * n_fbs: int
       - Number of filterbanks

    * sr: int
        - sampling rate. It is used to initialize ``freq_to_mel``.

    * init: str
        - if ``'mel'``, init with mel center frequencies and stds.

    * fmin: float
        - min frequency of filterbanks.
        - If `init == 'log'`, fmin should be > 0. Use `None` if you got no idea.

    * fmax: float
        - max frequency of filterbanks.
        - If `init == 'log'`, fmax is ignored.

    * trainable_fb: bool,
        - Whether the filterbanks are trainable or not.

    """

    def __init__(
            self,
            filterbank,
            data_format='default',
            **kwargs,
    ):
        self.filterbank = filterbank
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
        # x: 2d batch input
        output = K.dot(x, self.filterbank, axes=(self.freq_axis, 0))
        return output

    def get_config(self):
        config = {
            'filterbank': self.filterbank,
            'data_format': self.data_format,
        }
        return config


class Delta(Layer):
    """
    ### Delta

    ```python
    kapre.delta.Delta(win_length, mode, **kwargs)
    ```
    Calculates delta - local estimate of the derivative along time axis.
    See torchaudio.functional.compute_deltas or librosa.feature.delta for more details.

    #### Parameters

    * win_length: int
        - Window length of the derivative estimation
        - Default: 5

    * mode: str
        - It specifies pad mode of `tf.pad`. Case-insensitive
        - Default: 'symmetric'
        - {'symmetric', 'reflect', 'constant'}


    #### Returns

    A tensor with the same shape as input data.

    """

    def __init__(
            self, win_length=5, mode='symmetric', data_format='default', **kwargs
    ):

        assert data_format in ('default', 'channels_first', 'channels_last')
        assert win_length >= 3
        if win_length % 2 != 1:
            raise ValueError('win_length is expected to be an odd number, but it is %d' % win_length)
        assert mode.lower() in ('symmetric', 'reflect', 'constant')

        if data_format == 'default':
            self.data_format = K.image_data_format()
        else:
            self.data_format = data_format

        self.win_length = win_length
        self.mode = mode
        super(Delta, self).__init__(**kwargs)
        self.n = (self.win_length - 1) // 2  # half window length
        self.denom = 2 * sum([_n ** 2 for _n in range(1, self.n + 1, 1)])  # denominator

    def call(self, x):
        if self.data_format == 'channels_first':
            x = K.permute_dimensions(x, (0, 2, 3, 1))

        x = tf.pad(x, tf.constant([[0, 0], [0, 0], [self.n, self.n], [0, 0]]), mode=self.mode)  # pad over time
        kernel = K.arange(-self.n, self.n + 1, 1, dtype=K.floatx())
        kernel = K.reshape(kernel, (1, kernel.shape[-1], 1, 1))  # (freq, time)

        x = K.conv2d(x, kernel, 1, data_format='channels_last') / self.denom

        if self.data_format == 'channels_first':
            x = K.permute_dimensions(x, (0, 3, 1, 2))

        return x

    def get_config(self):
        config = {'win_length': self.win_length, 'mode': self.mode, 'data_format': self.data_format}

        return config
