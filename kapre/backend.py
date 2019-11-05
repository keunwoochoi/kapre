"""

Kapre backend functions
=======================\

|  Some backend functions that mainly use numpy.
|  Functions with Keras' backend is in ``backend_keras.py``.

Notes
-----
    * Don't forget to use ``K.float()``! Otherwise numpy uses float64.
    * Some functions are copied-and-pasted from librosa (to reduce dependency), but
        later I realised it'd be better to just use it.
    * TODO: remove copied code and use librosa.
"""
from tensorflow.keras import backend as K
import numpy as np
import librosa
# Forward compatability to replace xrange
from builtins import range

EPS = 1e-7


def eps():
    return EPS


def mel(sr, n_dft, n_mels=128, fmin=0.0, fmax=None, htk=False, norm=1):
    """[np] create a filterbank matrix to combine stft bins into mel-frequency bins
    use Slaney (said Librosa)

    n_mels: numbre of mel bands
    fmin : lowest frequency [Hz]
    fmax : highest frequency [Hz]
        If `None`, use `sr / 2.0`
    """
    return librosa.filters.mel(sr=sr, n_fft=n_dft, n_mels=n_mels,
                               fmin=fmin, fmax=fmax,
                               htk=htk, norm=norm).astype(K.floatx())


def get_stft_kernels(n_dft):
    """[np] Return dft kernels for real/imagnary parts assuming
        the input . is real.
    An asymmetric hann window is used (scipy.signal.hann).

    Parameters
    ----------
    n_dft : int > 0 and power of 2 [scalar]
        Number of dft components.

    Returns
    -------
        |  dft_real_kernels : np.ndarray [shape=(nb_filter, 1, 1, n_win)]
        |  dft_imag_kernels : np.ndarray [shape=(nb_filter, 1, 1, n_win)]

    * nb_filter = n_dft/2 + 1
    * n_win = n_dft

    """
    assert n_dft > 1 and ((n_dft & (n_dft - 1)) == 0), \
        ('n_dft should be > 1 and power of 2, but n_dft == %d' % n_dft)

    nb_filter = int(n_dft // 2 + 1)

    # prepare DFT filters
    timesteps = np.array(range(n_dft))
    w_ks = np.arange(nb_filter) * 2 * np.pi / float(n_dft)
    dft_real_kernels = np.cos(w_ks.reshape(-1, 1) * timesteps.reshape(1, -1))
    dft_imag_kernels = -np.sin(w_ks.reshape(-1, 1) * timesteps.reshape(1, -1))

    # windowing DFT filters
    dft_window = librosa.filters.get_window('hann', n_dft, fftbins=True)  # _hann(n_dft, sym=False)
    dft_window = dft_window.astype(K.floatx())
    dft_window = dft_window.reshape((1, -1))
    dft_real_kernels = np.multiply(dft_real_kernels, dft_window)
    dft_imag_kernels = np.multiply(dft_imag_kernels, dft_window)

    dft_real_kernels = dft_real_kernels.transpose()
    dft_imag_kernels = dft_imag_kernels.transpose()
    dft_real_kernels = dft_real_kernels[:, np.newaxis, np.newaxis, :]
    dft_imag_kernels = dft_imag_kernels[:, np.newaxis, np.newaxis, :]

    return dft_real_kernels.astype(K.floatx()), dft_imag_kernels.astype(K.floatx())


def filterbank_mel(sr, n_freq, n_mels=128, fmin=0.0, fmax=None, htk=False, norm=1):
    """[np] """
    return mel(sr, (n_freq - 1) * 2, n_mels=n_mels, fmin=fmin, fmax=fmax
               , htk=htk, norm=norm).astype(K.floatx())


def filterbank_log(sr, n_freq, n_bins=84, bins_per_octave=12,
                   fmin=None, spread=0.125):  # pragma: no cover
    """[np] Approximate a constant-Q filter bank for a fixed-window STFT.

    Each filter is a log-normal window centered at the corresponding frequency.

    Note: `logfrequency` in librosa 0.4 (deprecated), so copy-and-pasted,
        `tuning` was removed, `n_freq` instead of `n_fft`.

    Parameters
    ----------
    sr : number > 0 [scalar]
        audio sampling rate

    n_freq : int > 0 [scalar]
        number of frequency bins

    n_bins : int > 0 [scalar]
        Number of bins.  Defaults to 84 (7 octaves).

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave. Defaults to 12 (semitones).

    fmin : float > 0 [scalar]
        Minimum frequency bin. Defaults to `C1 ~= 32.70`

    spread : float > 0 [scalar]
        Spread of each filter, as a fraction of a bin.

    Returns
    -------
    C : np.ndarray [shape=(n_bins, 1 + n_fft/2)]
        log-frequency filter bank.
    """

    if fmin is None:
        fmin = 32.70319566

    # What's the shape parameter for our log-normal filters?
    sigma = float(spread) / bins_per_octave

    # Construct the output matrix
    basis = np.zeros((n_bins, n_freq))

    # Get log frequencies of bins
    log_freqs = np.log2(librosa.fft_frequencies(sr, (n_freq - 1) * 2)[1:])

    for i in range(n_bins):
        # What's the center (median) frequency of this filter?
        c_freq = fmin * (2.0 ** (float(i) / bins_per_octave))

        # Place a log-normal window around c_freq
        basis[i, 1:] = np.exp(-0.5 * ((log_freqs - np.log2(c_freq)) / sigma) ** 2
                              - np.log2(sigma) - log_freqs)

    # Normalize the filters
    basis = librosa.util.normalize(basis, norm=1, axis=1)

    return basis.astype(K.floatx())
