"""Backend operations of Kapre.

This module summarizes operations and functions that are used in Kapre layers.

Attributes:
    _CH_FIRST_STR (str): 'channels_first', a pre-defined string.
    _CH_LAST_STR (str): 'channels_last', a pre-defined string.
    _CH_DEFAULT_STR (str): 'default', a pre-defined string.

"""
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import librosa

_CH_FIRST_STR = 'channels_first'
_CH_LAST_STR = 'channels_last'
_CH_DEFAULT_STR = 'default'


def validate_data_format_str(data_format):
    """A function that validates the data format string."""
    if data_format not in (_CH_DEFAULT_STR, _CH_FIRST_STR, _CH_LAST_STR):
        raise ValueError(
            'data_format should be one of {}'.format(
                str([_CH_FIRST_STR, _CH_LAST_STR, _CH_DEFAULT_STR])
            )
            + ' but we received {}'.format(data_format)
        )


def magnitude_to_decibel(x, ref_value=1.0, amin=1e-5, dynamic_range=80.0):
    """A function that converts magnitude to decibel scaling.
    In essence, it runs `10 * log10(x)`, but with some other utility operations.

    Similar to `librosa.amplitude_to_db` with `ref=1.0` and `top_db=dynamic_range`

    Args:
        x (`Tensor`): float tensor. Can be batch or not. Something like magnitude of STFT.
        ref_value (`float`): an input value that would become 0 dB in the result.
            For spectrogram magnitudes, ref_value=1.0 usually make the decibel-sclaed output to be around zero
            if the input audio was in [-1, 1].
        amin (`float`): the noise floor of the input. An input that is smaller than `amin`, it's converted to `amin`.
        dynamic_range (`float`): range of the resulting value. E.g., if the maximum magnitude is 30 dB,
            the noise floor of the output would become (30 - dynamic_range) dB

    Returns:
        log_spec (`Tensor`): a decibel-scaled version of `x`.

    Notes:
        In many deep learning based application, the input spectrogram magnitudes (e.g., abs(STFT)) are decibel-scaled
        (=logarithmically mapped) for a better performance.

    Example:
        ::

            input_shape = (2048, 1)  # mono signal
            model = Sequential()
            model.add(kapre.Frame(frame_length=1024, hop_length=512, input_shape=input_shape))
            # now the shape is (batch, n_frame=3, frame_length=1024, ch=1)

    """

    def _log10(x):
        return tf.math.log(x) / tf.math.log(tf.constant(10, dtype=x.dtype))

    if K.ndim(x) > 1:  # we assume x is batch in this case
        max_axis = tuple(range(K.ndim(x))[1:])
    else:
        max_axis = None

    if amin is None:
        amin = 1e-5

    log_spec = 10.0 * _log10(tf.math.maximum(x, amin))
    log_spec = log_spec - 10.0 * _log10(tf.math.maximum(amin, ref_value))

    log_spec = tf.math.maximum(
        log_spec, tf.math.reduce_max(log_spec, axis=max_axis, keepdims=True) - dynamic_range
    )

    return log_spec


def filterbank_mel(
    sample_rate, n_freq, n_mels=128, f_min=0.0, f_max=None, htk=False, norm='slaney'
):
    """A wrapper for librosa.filters.mel that additionally does transpose and tensor conversion

    Args:
        sample_rate (`int`): sample rate of the input audio
        n_freq (`int`): number of frequency bins in the input STFT magnitude.
        n_mels (`int`): the number of mel bands
        f_min (`float`): lowest frequency that is going to be included in the mel filterbank (Hertz)
        f_max (`float`): highest frequency that is going to be included in the mel filterbank (Hertz)
        htk (bool): whether to use `htk` formula or not
        norm: The default, 'slaney', would normalize the the mel weights by the width of the mel band.

    Returns:
        (`Tensor`): mel filterbanks. Shape=`(n_freq, n_mels)`
    """
    filterbank = librosa.filters.mel(
        sr=sample_rate,
        n_fft=(n_freq - 1) * 2,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        htk=htk,
        norm=norm,
    ).astype(K.floatx())
    return tf.convert_to_tensor(filterbank.T)


def filterbank_log(sample_rate, n_freq, n_bins=84, bins_per_octave=12, f_min=None, spread=0.125):
    """A function that returns a approximation of constant-Q filter banks for a fixed-window STFT.
    Each filter is a log-normal window centered at the corresponding frequency.

    Args:
        sample_rate (`int`): audio sampling rate
        n_freq (`int`): number of the input frequency bins. E.g., `n_fft / 2 + 1`
        n_bins (`int`): number of the resulting log-frequency bins.  Defaults to 84 (7 octaves).
        bins_per_octave (`int`): number of bins per octave. Defaults to 12 (semitones).
        f_min (`float`): lowest frequency that is going to be included in the log filterbank. Defaults to `C1 ~= 32.70`
        spread (`float`): spread of each filter, as a fraction of a bin.

    Returns:
        (`Tensor`): log-frequency filterbanks. Shape=`(n_freq, n_bins)`

    Note:
        The code is originally from `logfrequency` in librosa 0.4 (deprecated) and copy-and-pasted.
        `tuning` parameter was removed and we use `n_freq` instead of `n_fft`.
    """

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
    log_freqs = np.log2(librosa.fft_frequencies(sample_rate, (n_freq - 1) * 2)[1:])

    for i in range(n_bins):
        # What's the center (median) frequency of this filter?
        c_freq = f_min * (2.0 ** (float(i) / bins_per_octave))

        # Place a log-normal window around c_freq
        basis[i, 1:] = np.exp(
            -0.5 * ((log_freqs - np.log2(c_freq)) / sigma) ** 2 - np.log2(sigma) - log_freqs
        )

    # Normalize the filters
    basis = librosa.util.normalize(basis, norm=1, axis=1)
    basis = basis.astype(K.floatx())

    return tf.convert_to_tensor(basis.T)


def mu_law_encoding(signal, quantization_channels):
    """Encode signal based on mu-law companding. Also called mu-law compressing.

    This algorithm assumes the signal has been scaled to between -1 and 1 and returns a signal encoded
    with values from 0 to quantization_channels - 1.
    See `Wikipedia <https://en.wikipedia.org/wiki/Μ-law_algorithm>`_ for more details.

    Args:
        signal (float `Tensor`): audio signal to encode
        quantization_channels (positive int): Number of channels. For 8-bit encoding, use 256.

    Returns:
        signal_mu (int `Tensor`): mu-encoded signal
    """
    mu = quantization_channels - 1.0
    signal_mu = tf.math.sign(signal) * tf.math.log1p(mu * tf.math.abs(signal)) / tf.math.log1p(mu)
    signal_mu = tf.cast(((signal_mu + 1) / 2.0 * mu + 0.5), tf.int32)
    return signal_mu


def mu_law_decoding(signal_mu, quantization_channels):
    """Decode mu-law encoded signals based on mu-law companding. Also called mu-law expanding.

    See `Wikipedia <https://en.wikipedia.org/wiki/Μ-law_algorithm>`_ for more details.

    Args:
        signal_mu (int `Tensor`): mu-encoded signal to decode
        quantization_channels (positive int): Number of channels. For 8-bit encoding, use 256.

    Returns:
        signal (float `Tensor`): decoded audio signal
    """
    mu = quantization_channels - 1.0
    signal_mu = K.cast_to_floatx(signal_mu)

    signal = (signal_mu / mu) * 2 - 1.0
    signal = (
        tf.math.sign(signal) * (tf.math.exp(tf.math.abs(signal) * tf.math.log1p(mu)) - 1.0) / mu
    )
    return signal
