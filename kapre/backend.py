"""
Kapre backend functions
=======================

|  Some backend functions that mainly use numpy.
|  Functions with Keras' backend is in ``backend_keras.py``.

Notes
-----
    * Don't forget to use ``K.float()``! Otherwise numpy uses float64.
    * Some functions are copied-and-pasted from librosa (to reduce dependency), but
        later I realised it'd be better to just use it.
"""
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import librosa

if K.image_data_format() == 'channels_first':
    CH_AXIS = 1
else:
    CH_AXIS = 3


def amplitude_to_decibel(x, amin=None, dynamic_range=120.0):
    """[K] Convert (linear) amplitude to decibel (log10(x)).

    Parameters
    ----------
    x: Keras *batch* tensor or variable.

    amin: minimum amplitude. amplitude smaller than `amin` is set to this.

    dynamic_range: dynamic_range in decibel

    """
    if amin is None:
        amin = K.epsilon()

    log_spec = 10 * tf.math.log(tf.maximum(x, amin)) / tf.math.log(tf.constant(10, dtype=x.dtype))
    if K.ndim(x) > 1:
        axis = tuple(range(K.ndim(x))[1:])
    else:
        axis = None

    log_spec = log_spec - tf.math.reduce_max(log_spec, axis=axis, keepdims=True)  # [-?, 0]
    log_spec = tf.math.maximum(log_spec, -1 * dynamic_range)  # [-120, 0]
    return log_spec


def filterbank_mel(sample_rate, n_freq, n_mels=128, f_min=0.0, f_max=None, htk=False, norm='slaney'):
    """A wrapper for librosa.filters.mel that additionally does transpose and tensor conversion

    Args:
        n_mels: numbre of mel bands
        f_min : lowest frequency [Hz]
        f_max : highest frequency [Hz]

    Return:
        filterbank (tensor): (n_freq, n_mels)
    """
    filterbank = librosa.filters.mel(
        sr=sample_rate, n_fft=(n_freq - 1) * 2, n_mels=n_mels, fmin=f_min, fmax=f_max, htk=htk, norm=norm
    ).astype(K.floatx())
    return tf.convert_to_tensor(filterbank.T)


def filterbank_log(
        sample_rate, n_freq, n_bins=84, bins_per_octave=12, f_min=None, spread=0.125
):
    """Approximate a constant-Q filter bank for a fixed-window STFT.

    Each filter is a log-normal window centered at the corresponding frequency.

    Note: `logfrequency` in librosa 0.4 (deprecated), so copy-and-pasted,
        `tuning` was removed, `n_freq` instead of `n_fft`.

    Parameters
    ----------
    sample_rate : number > 0 [scalar]
        audio sampling rate

    n_freq : int > 0 [scalar]
        number of frequency bins

    n_bins : int > 0 [scalar]
        Number of bins.  Defaults to 84 (7 octaves).

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave. Defaults to 12 (semitones).

    f_min : float > 0 [scalar]
        Minimum frequency bin. Defaults to `C1 ~= 32.70`

    spread : float > 0 [scalar]
        Spread of each filter, as a fraction of a bin.

    Returns
    -------
        C : tensor[shape=(n_freq, n_bins)]
            log-frequency filter bank.

    """

    if f_min is None:
        f_min = 32.70319566

    f_max = f_min * 2 ** (n_bins / bins_per_octave)
    if f_max > sample_rate // 2:
        raise RuntimeError('Maximum frequency of log filterbank should be lower or equal to the maximum'
                           'frequency of the input, but f_max=%f and maximum frequency is %f. '
                           'Fix it by reducing n_bins, increasing bins_per_octave, reduce f_min.\n'
                           'You can also do it by increasing sample_rate but it means you need to upsample'
                           'the input data, too.' % (f_max, sample_rate))

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
