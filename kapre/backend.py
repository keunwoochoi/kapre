"""Backend operations of Kapre.

This module summarizes operations and functions that are used in Kapre layers.

Attributes:
    _CH_FIRST_STR (str): 'channels_first', a pre-defined string.
    _CH_LAST_STR (str): 'channels_last', a pre-defined string.
    _CH_DEFAULT_STR (str): 'default', a pre-defined string.

"""
from __future__ import annotations

from typing import Optional, Union, Callable, Any
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

# librosa is imported inside functions to avoid import errors during type checking

# Compatibility layer for different TensorFlow versions
def _get_image_data_format() -> str:
    """Get image data format in a version-compatible way.

    Returns:
        str: The image data format ('channels_first' or 'channels_last')
    """
    try:
        # Try newer Keras config API first (TensorFlow 2.13+)
        import keras.config
        return keras.config.image_data_format
    except (ImportError, AttributeError):
        try:
            # Try older Keras backend API
            return K.image_data_format()
        except AttributeError:
            # Fallback to channels_last for newer TF versions
            return 'channels_last'


def _get_floatx() -> str:
    """Get the default float type in a version-compatible way.

    Returns:
        str: The default float type (e.g., 'float32', 'float64')
    """
    try:
        # Try newer Keras backend API
        return K.floatx()
    except AttributeError:
        # Fallback for newer TF versions
        return tf.keras.backend.floatx()

_CH_FIRST_STR = 'channels_first'
_CH_LAST_STR = 'channels_last'
_CH_DEFAULT_STR = 'default'


def get_window_fn(window_name: Optional[str] = None) -> Callable[[int], tf.Tensor]:
    """Return a window function given its name.

    This function is used inside layers such as `STFT` to get a window function.

    Args:
        window_name: Name of window function. On Tensorflow 2.3, there are five windows available in
        `tf.signal` (`hamming_window`, `hann_window`, `kaiser_bessel_derived_window`, `kaiser_window`, `vorbis_window`).
        If None, defaults to 'hann_window'.

    Returns:
        Callable that takes window length and returns a window tensor.

    Raises:
        NotImplementedError: If the window name is not supported.
    """

    if window_name is None:
        return tf.signal.hann_window

    available_windows = {
        'hamming_window': tf.signal.hamming_window,
        'hann_window': tf.signal.hann_window,
    }
    if hasattr(tf.signal, 'kaiser_bessel_derived_window'):
        available_windows['kaiser_bessel_derived_window'] = tf.signal.kaiser_bessel_derived_window
    if hasattr(tf.signal, 'kaiser_window'):
        available_windows['kaiser_window'] = tf.signal.kaiser_window
    if hasattr(tf.signal, 'vorbis_window'):
        available_windows['vorbis_window'] = tf.signal.vorbis_window

    if window_name not in available_windows:
        raise NotImplementedError(
            'Window name %s is not supported now. Currently, %d windows are'
            'supported - %s'
            % (
                window_name,
                len(available_windows),
                ', '.join([k for k in available_windows.keys()]),
            )
        )

    return available_windows[window_name]


def validate_data_format_str(data_format: str) -> None:
    """Validate the data format string.

    Args:
        data_format: The data format string to validate. Must be one of 'default',
            'channels_first', or 'channels_last'.

    Raises:
        TypeError: If data_format is not a string.
        ValueError: If the data format is not one of the supported values.
    """
    if not isinstance(data_format, str):
        raise TypeError(
            f'data_format must be a string, got {type(data_format).__name__}: {data_format}'
        )

    if data_format not in (_CH_DEFAULT_STR, _CH_FIRST_STR, _CH_LAST_STR):
        raise ValueError(
            f'data_format must be one of {[_CH_FIRST_STR, _CH_LAST_STR, _CH_DEFAULT_STR]}, '
            f'got: {data_format!r}'
        )


def magnitude_to_decibel(
    x: tf.Tensor,
    ref_value: float = 1.0,
    amin: float = 1e-5,
    dynamic_range: float = 80.0,
) -> tf.Tensor:
    """Convert magnitude to decibel scaling.

    In essence, it runs `10 * log10(x)`, but with some other utility operations.

    Similar to `librosa.power_to_db` with `ref=1.0` and `top_db=dynamic_range`

    Args:
        x: Float tensor. Can be batch or not. Something like magnitude of STFT.
        ref_value: An input value that would become 0 dB in the result.
            For spectrogram magnitudes, ref_value=1.0 usually make the decibel-scaled output to be around zero
            if the input audio was in [-1, 1]. Must be positive.
        amin: The noise floor of the input. An input that is smaller than `amin`, it's converted to `amin`.
            Must be positive. Defaults to 1e-5.
        dynamic_range: Range of the resulting value. E.g., if the maximum magnitude is 30 dB,
            the noise floor of the output would become (30 - dynamic_range) dB. Must be positive.

    Returns:
        A decibel-scaled version of `x`.

    Raises:
        ValueError: If any parameter has an invalid value.

    Note:
        In many deep learning based application, the input spectrogram magnitudes (e.g., abs(STFT)) are decibel-scaled
        (=logarithmically mapped) for a better performance.

    Example:
        ::

            input_shape = (2048, 1)  # mono signal
            model = Sequential()
            model.add(kapre.Frame(frame_length=1024, hop_length=512, input_shape=input_shape))
            # now the shape is (batch, n_frame=3, frame_length=1024, ch=1)

    """
    # Input validation
    if ref_value <= 0:
        raise ValueError(f'ref_value must be positive, got: {ref_value}')
    if amin <= 0:
        raise ValueError(f'amin must be positive, got: {amin}')
    if dynamic_range <= 0:
        raise ValueError(f'dynamic_range must be positive, got: {dynamic_range}')

    def _log10(x):
        return tf.math.log(x) / tf.math.log(tf.constant(10, dtype=x.dtype))

    if K.ndim(x) > 1:  # we assume x is batch in this case
        max_axis = tuple(range(K.ndim(x))[1:])
    else:
        max_axis = None

    if amin is None:
        amin = 1e-5

    amin = tf.cast(amin, dtype=x.dtype)
    log_spec = 10.0 * _log10(tf.math.maximum(x, amin))
    log_spec = log_spec - 10.0 * _log10(tf.math.maximum(amin, ref_value))

    log_spec = tf.math.maximum(
        log_spec, tf.math.reduce_max(log_spec, axis=max_axis, keepdims=True) - dynamic_range
    )

    return log_spec


def filterbank_mel(
    sample_rate: int,
    n_freq: int,
    n_mels: int = 128,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    htk: bool = False,
    norm: Union[str, int] = 'slaney',
) -> tf.Tensor:
    """A wrapper for librosa.filters.mel that additionally does transpose and tensor conversion

    Args:
        sample_rate: Sample rate of the input audio
        n_freq: Number of frequency bins in the input STFT magnitude.
        n_mels: The number of mel bands
        f_min: Lowest frequency that is going to be included in the mel filterbank (Hertz)
        f_max: Highest frequency that is going to be included in the mel filterbank (Hertz)
        htk: Whether to use `htk` formula or not
        norm: The default, 'slaney', would normalize the the mel weights by the width of the mel band.

    Returns:
        Mel filterbanks. Shape=(n_freq, n_mels)
    """
    import librosa

    filterbank = librosa.filters.mel(  # type: ignore
        sr=sample_rate,
        n_fft=(n_freq - 1) * 2,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        htk=htk,
        norm=norm,
    ).astype(_get_floatx())
    return tf.convert_to_tensor(filterbank.T)


def filterbank_log(
    sample_rate: int,
    n_freq: int,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    f_min: Optional[float] = None,
    spread: float = 0.125,
) -> tf.Tensor:
    """Return an approximation of constant-Q filter banks for a fixed-window STFT.

    Each filter is a log-normal window centered at the corresponding frequency.

    Args:
        sample_rate: Audio sampling rate
        n_freq: Number of the input frequency bins. E.g., `n_fft / 2 + 1`
        n_bins: Number of the resulting log-frequency bins. Defaults to 84 (7 octaves).
        bins_per_octave: Number of bins per octave. Defaults to 12 (semitones).
        f_min: Lowest frequency that is going to be included in the log filterbank. Defaults to `C1 ~= 32.70`
        spread: Spread of each filter, as a fraction of a bin.

    Returns:
        Log-frequency filterbanks. Shape=(n_freq, n_bins)

    Note:
        The code is originally from `logfrequency` in librosa 0.4 (deprecated) and copy-and-pasted.
        `tuning` parameter was removed and we use `n_freq` instead of `n_fft`.
    """
    import librosa

    if f_min is None:
        f_min = 32.70319566

    f_max = f_min * 2 ** (n_bins / bins_per_octave)
    if f_max > sample_rate // 2:
        raise RuntimeError(
            'Maximum frequency of log filterbank should be lower or equal to the maximum'
            'frequency of the input (defined by its sample rate), '
            'but f_max=%f and maximum frequency is %f. \n'
            'Fix it by reducing n_bins, increasing bins_per_octave and/or reducing f_min.\n'
            'You can also do it by increasing sample_rate but it means you need to upsample'
            'the input audio data, too.' % (f_max, sample_rate)
        )

    # What's the shape parameter for our log-normal filters?
    sigma = float(spread) / bins_per_octave

    # Construct the output matrix
    basis = np.zeros((n_bins, n_freq))

    # Get log frequencies of bins
    log_freqs = np.log2(librosa.fft_frequencies(sr=sample_rate, n_fft=(n_freq - 1) * 2)[1:])

    for i in range(n_bins):
        # What's the center (median) frequency of this filter?
        c_freq = f_min * (2.0 ** (float(i) / bins_per_octave))

        # Place a log-normal window around c_freq
        basis[i, 1:] = np.exp(
            -0.5 * ((log_freqs - np.log2(c_freq)) / sigma) ** 2 - np.log2(sigma) - log_freqs
        )

    # Normalize the filters
    basis = librosa.util.normalize(basis, norm=1, axis=1)  # type: ignore
    basis = basis.astype(_get_floatx())

    return tf.convert_to_tensor(basis.T)


def mu_law_encoding(signal: tf.Tensor, quantization_channels: int) -> tf.Tensor:
    """Encode signal based on mu-law companding. Also called mu-law compressing.

    This algorithm assumes the signal has been scaled to between -1 and 1 and returns a signal encoded
    with values from 0 to quantization_channels - 1.
    See `Wikipedia <https://en.wikipedia.org/wiki/Μ-law_algorithm>`_ for more details.

    Args:
        signal: Audio signal to encode
        quantization_channels: Number of channels. For 8-bit encoding, use 256.

    Returns:
        Mu-encoded signal
    """
    mu = quantization_channels - 1.0
    signal_mu = tf.math.sign(signal) * tf.math.log1p(mu * tf.math.abs(signal)) / tf.math.log1p(mu)
    signal_mu = tf.cast(((signal_mu + 1) / 2.0 * mu + 0.5), tf.int32)
    return signal_mu


def mu_law_decoding(signal_mu: tf.Tensor, quantization_channels: int) -> tf.Tensor:
    """Decode mu-law encoded signals based on mu-law companding. Also called mu-law expanding.

    See `Wikipedia <https://en.wikipedia.org/wiki/Μ-law_algorithm>`_ for more details.

    Args:
        signal_mu: Mu-encoded signal to decode
        quantization_channels: Number of channels. For 8-bit encoding, use 256.

    Returns:
        Decoded audio signal
    """
    mu = quantization_channels - 1.0
    signal_mu = K.cast_to_floatx(signal_mu)

    signal = (signal_mu / mu) * 2 - 1.0
    signal = (
        tf.math.sign(signal) * (tf.math.exp(tf.math.abs(signal) * tf.math.log1p(mu)) - 1.0) / mu
    )
    return signal
