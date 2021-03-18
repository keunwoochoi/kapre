"""Tflite compatible versions of Kapre layers.

`STFTTflite` is a tflite compatible version of `STFT`. Tflite does not support complex
types, thus real and imaginary parts are returned as an extra (last) dimension.
Ouput shape is now: `(batch, channel, time, re/im)` or `(batch, time, channel, re/im)`.

Because of the change of dimension, Tflite compatible layers are provided to
process the resulting STFT; `MagnitudeTflite` and `PhaseTflite` are layers that
calculate the magnitude and phase respectively from the output of `STFTTflite`.
"""
import tensorflow as tf
from .backend import _CH_FIRST_STR, _CH_LAST_STR, _CH_DEFAULT_STR
from .tflite_compatible_stft import stft_tflite, atan2_tflite

# import non-tflite compatible layers to inheret from.
from .time_frequency import STFT, InverseSTFT, Magnitude, Phase


__all__ = [
    'STFTTflite',
    # 'InverseSTFTTflite',  # NOTE (PK): todo
    'MagnitudeTflite',
    'PhaseTflite',
]


class STFTTflite(STFT):
    """
    A Short-time Fourier transform layer (tflite compatible).

    Ues `stft_tflite` from tflite_compatible_stft.py, this contains a tflite
    compatible stft (using a rdft), and `fixed_frame()` to window the audio.
    Tflite does not cope with comple types so real and imaginary parts are stored in extra dim.
    Ouput shape is now: (batch, channel, time, re/im) or (batch, time, channel, re/im)

    Additionally, it reshapes the output to be a proper 2D batch.

    If `output_data_format == 'channels_last'`, the output shape is `(batch, time, freq, channel, re/imag)`
    If `output_data_format == 'channels_first'`, the output shape is `(batch, channel, time, freq, re/imag)`

    Args:
        n_fft (int): Number of FFTs. Defaults to `2048`
        win_length (int or None): Window length in sample. Defaults to `n_fft`.
        hop_length (int or None): Hop length in sample between analysis windows. Defaults to `n_fft // 4` following Librosa.
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
            model = Sequential()  # tflite compatible model
            model.add(kapre.STFTTflite(n_fft=1024, hop_length=512, input_shape=input_shape))
            # now the shape is (batch, n_frame=3, n_freq=513, ch=1, re/im=2)
            # and the dtype is real

    """

    def call(self, x):
        """
        Compute STFT of the input signal. If the `time` axis is not the last axis of `x`, it should be transposed first.

        Args:
            x (float `Tensor`): batch of audio signals, (batch, ch, time) or (batch, time, ch) based on input_data_format

        Return:
            (real `Tensor`): A STFT representation of x in a 2D batch shape. The last dimension is size two and contains
            the real and imaginary parts of the stft.
            Its shape is (batch, time, freq, ch, 2) or (batch. ch, time, freq, 2) depending on `output_data_format` and
            `time` is the number of frames, which is `((len_src + (win_length - hop_length) / hop_length) // win_length )`
            if `pad_end` is `True`. `freq` is the number of fft unique bins, which is `n_fft // 2 + 1` (the unique components of the FFT).
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
        stfts = stft_tflite(
            waveforms,
            frame_length=self.win_length,
            frame_step=self.hop_length,
            fft_length=self.n_fft,
            window_fn=self.window_fn,
            pad_end=self.pad_end,
        )  # (batch, ch, time, freq, re/imag)

        if self.output_data_format == _CH_LAST_STR:
            # tflite compatible stft produces real and imag in 1st dim
            stfts = tf.transpose(stfts, perm=(0, 2, 3, 1, 4))  # (batch, t, f, ch, re/im)

        return stfts


class MagnitudeTflite(Magnitude):
    """Compute the magnitude of the input (tflite compatible).

    The input is a real tensor, the last dimension has a size of `2`
    representing real and imaginary parts respectively.

    Example:
        ::

            input_shape = (2048, 1)  # mono signal
            model = Sequential()
            model.add(kapre.STFTTflite(n_fft=1024, hop_length=512, input_shape=input_shape))
            mode.add(MagnitudeTflite())
            # now the shape is (batch, n_frame=3, n_freq=513, ch=1) and dtype is float

    """

    def call(self, x):
        """
        Args:
            x (real or complex `Tensor`): input is real tensor whose last
                dimension has a size of `2` and represents real and imaginary
                parts

        Returns:
            (float `Tensor`): magnitude of `x`
        """
        return tf.norm(x, ord='euclidean', axis=-1)


class PhaseTflite(Phase):
    """Compute the phase of the complex input in radian, resulting in a float tensor (tflite compatible).

    Note TF lite does not natively support atan, used in tf.math.angle, so an
    approximation is provided. You may want to use this approximation if you
    generate data using a non-tf-lite compatible STFT (faster) but want the same
    approximations in the training data.

    Args:
        approx_atan_accuracy (`int`): if `None` will use `tf.math.angle()` to
            calculate the phase accurately. If an `int` this is the number of
            iterations to calculate the approximate `atan()` using a tflite compatible
            method. the higher the number the more accurate e.g.
            `approx_atan_accuracy=29000`. You may want to experiment with adjusting
            this number: trading off accuracy with inference speed.

    Example:
        ::

            input_shape = (2048, 1)  # mono signal
            model = Sequential()
            model.add(kapre.STFTTflite(n_fft=1024, hop_length=512, input_shape=input_shape))
            model.add(PhaseTflite(approx_atan_accuracy=5000))
            # now the shape is (batch, n_frame=3, n_freq=513, ch=1) and dtype is float

    """

    def call(self, x):
        """
        Args:
            x (real): input is real tensor with five
                dimensions (last dim is re/imag)

        Returns:
            (float `Tensor`): phase of `x` (Radian)
        """
        return atan2_tflite(x[:, :, :, :, 1], x[:, :, :, :, 0], n=self.approx_atan_accuracy)
