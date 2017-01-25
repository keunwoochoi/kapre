''' Some backend computation with numpy.
Similar computation with keras is on `backend_keras.py`.

* Don't forget to use K.float()! Otherwise numpy uses float64. 
* Some functions are copied-and-pasted from librosa (to reduce dependency), but
    later I realised it'd be better to just use it. 
    TODO: remove copied code and use librosa.
'''
from keras import backend as K
import numpy as np
from librosa.core.time_frequency import fft_frequencies, mel_frequencies
import librosa

TOL = 1e-5
EPS = 1e-7


def tolerance():
    return TOL


def eps():
    return EPS


def log_frequencies(n_bins=128, fmin=None, fmax=11025.0):
    """[np] Compute the center frequencies of bands
    TODO: ...do I use it?
    """
    if fmin is None:
        fmin = librosa.core.time_frequency.note_to_hz('C1').astype(K.floatx())
    return np.logspace(np.log2(fmin), np.log2(fmax), num=n_bins, base=2).astype(K.floatx())


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0):
    """[np] Compute the center frequencies of mel bands.
    `htk` is removed.
    
    Keunwoo: copied from Librosa
    """

    def _mel_to_hz(mels):
        """Convert mel bin numbers to frequencies
        copied from Librosa
        """
        mels = np.atleast_1d(mels)

        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # And now the nonlfinear scale
        min_log_hz = 1000.0  # beginning of log region
        min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
        logstep = np.log(6.4) / 27.0  # step size for log region
        log_t = (mels >= min_log_mel)

        freqs[log_t] = min_log_hz \
                       * np.exp(logstep * (mels[log_t] - min_log_mel))

        return freqs

    def _hz_to_mel(frequencies):
        """Convert Hz to Mels
        
        Keunwoo: copied from Librosa        
        """
        frequencies = np.atleast_1d(frequencies)

        # Fill in the linear part
        f_min = 0.0
        f_sp = 200.0 / 3

        mels = (frequencies - f_min) / f_sp

        # Fill in the log-scale part
        min_log_hz = 1000.0  # beginning of log region
        min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
        logstep = np.log(6.4) / 27.0  # step size for log region

        log_t = (frequencies >= min_log_hz)
        mels[log_t] = min_log_mel \
                      + np.log(frequencies[log_t] / min_log_hz) / logstep

        return mels

    ''' mel_frequencies body starts '''
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mels = _hz_to_mel(fmin)
    max_mel = _hz_to_mel(fmax)

    mels = np.linspace(min_mels, max_mel, n_mels)

    return _mel_to_hz(mels).astype(K.floatx())


def _dft_frequencies(sr=22050, n_dft=2048):
    '''[np] Alternative implementation of `np.fft.fftfreqs` (said Librosa)
    copied from Librosa

    '''
    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_dft // 2),
                       endpoint=True).astype(K.floatx())


def mel(sr, n_dft, n_mels=128, fmin=0.0, fmax=None):
    '''[np] create a filterbank matrix to combine stft bins into mel-frequency bins
    use Slaney
    Keunwoo: copied from Librosa, librosa.filters.mel
    
    n_mels: numbre of mel bands
    fmin : lowest frequency [Hz]
    fmax : highest frequency [Hz]
        If `None`, use `sr / 2.0`
    '''
    if fmax is None:
        fmax = float(sr) / 2

    # init
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_dft // 2)))

    # center freqs of each FFT bin
    dftfreqs = _dft_frequencies(sr=sr, n_dft=n_dft)

    # centre freqs of mel bands
    freqs = mel_frequencies(n_mels + 2,
                            fmin=fmin,
                            fmax=fmax)
    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (freqs[2:n_mels + 2] - freqs[:n_mels])

    for i in range(n_mels):
        # lower and upper slopes qfor all bins
        lower = (dftfreqs - freqs[i]) / (freqs[i + 1] - freqs[i])
        upper = (freqs[i + 2] - dftfreqs) / (freqs[i + 2] - freqs[i + 1])

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper)) * enorm[i]

    return weights.astype(K.floatx())


def get_stft_kernels(n_dft, keras_ver='new'):
    '''[np] Return dft kernels for real/imagnary parts assuming
        the input . is real.
    An asymmetric hann window is used (scipy.signal.hann).

    Parameters
    ----------
    n_dft : int > 0 and power of 2 [scalar]
        Number of dft components.

    keras_ver : string, 'new' or 'old'
        It determines the reshaping strategy.

    Returns
    -------
    dft_real_kernels : np.ndarray [shape=(nb_filter, 1, 1, n_win)]
    dft_imag_kernels : np.ndarray [shape=(nb_filter, 1, 1, n_win)]

    * nb_filter = n_dft/2 + 1
    * n_win = n_dft

    '''
    assert n_dft > 1 and ((n_dft & (n_dft - 1)) == 0), \
        ('n_dft should be > 1 and power of 2, but n_dft == %d' % n_dft)

    nb_filter = n_dft / 2 + 1
    dtype = K.floatx()

    # prepare DFT filters
    timesteps = range(n_dft)
    w_ks = [(2 * np.pi * k) / float(n_dft) for k in xrange(n_dft)]
    dft_real_kernels = np.array([[np.cos(w_k * n) for n in timesteps]
                                 for w_k in w_ks])
    dft_imag_kernels = np.array([[np.sin(w_k * n) for n in timesteps]
                                 for w_k in w_ks])

    # windowing DFT filters
    dft_window = _hann(n_dft, sym=False)
    dft_window = dft_window.reshape((1, -1))
    dft_real_kernels = np.multiply(dft_real_kernels, dft_window)
    dft_imag_kernels = np.multiply(dft_imag_kernels, dft_window)

    if keras_ver == 'old':  # 1.0.6: reshape filter e.g. (5, 8) -> (5, 1, 8, 1)
        dft_real_kernels = dft_real_kernels[:nb_filter]
        dft_imag_kernels = dft_imag_kernels[:nb_filter]
        dft_real_kernels = dft_real_kernels[:, np.newaxis, :, np.newaxis]
        dft_imag_kernels = dft_imag_kernels[:, np.newaxis, :, np.newaxis]
    else:
        dft_real_kernels = dft_real_kernels[:nb_filter].transpose()
        dft_imag_kernels = dft_imag_kernels[:nb_filter].transpose()
        dft_real_kernels = dft_real_kernels[:, np.newaxis, np.newaxis, :]
        dft_imag_kernels = dft_imag_kernels[:, np.newaxis, np.newaxis, :]

    return dft_real_kernels.astype(K.floatx()), dft_imag_kernels.astype(K.floatx())


def _hann(M, sym=True):
    '''[np] 
    Return a Hann window.
    copied and pasted from scipy.signal.hann,
    https://github.com/scipy/scipy/blob/v0.14.0/scipy/signal/windows.py#L615
    ----------

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.
    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).
    
    '''
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, 'd')
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = np.arange(0, M)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (M - 1))
    if not sym and not odd:
        w = w[:-1]
    return w.astype(K.floatx())


# Filterbanks
def filterbank_mel(sr, n_freq, n_mels=128, fmin=0.0, fmax=None):
    '''[np] '''
    return mel(sr, (n_freq - 1) * 2, n_mels=128, fmin=0.0, fmax=None).astype(K.floatx())


def filterbank_log(sr, n_freq, n_bins=84, bins_per_octave=12,
                   fmin=None, spread=0.125):  # pragma: no cover
    '''[np] Approximate a constant-Q filter bank for a fixed-window STFT.

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
    '''

    if fmin is None:
        fmin = librosa.core.time_frequency.note_to_hz('C1')

    # What's the shape parameter for our log-normal filters?
    sigma = float(spread) / bins_per_octave

    # Construct the output matrix
    basis = np.zeros((n_bins, n_freq))

    # Get log frequencies of bins
    log_freqs = np.log2(fft_frequencies(sr, (n_freq - 1) * 2)[1:])

    for i in range(n_bins):
        # What's the center (median) frequency of this filter?
        c_freq = fmin * (2.0 ** (float(i) / bins_per_octave))

        # Place a log-normal window around c_freq
        basis[i, 1:] = np.exp(-0.5 * ((log_freqs - np.log2(c_freq)) / sigma) ** 2
                              - np.log2(sigma) - log_freqs)

    # Normalize the filters
    basis = librosa.util.normalize(basis, norm=1, axis=1)

    return basis.astype(K.floatx())
