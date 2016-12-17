from keras import backend as K
import numpy as np


def amplitude_to_decibel(x, ref_power=1.0, amin=1e-10, top_db=80.0):
    assert isinstance(ref_power, float) or ref_power == 'max'
        
    log_spec = 10 * K.log(K.maximum(x, amin)) / K.log(10)
    # if ref_power == 'max':
    log_spec = log_spec - K.max(log_spec)  # [-?, 0]
    # else:
    #     log_spec = log_spec - ref_power
    log_spec = K.maximum(log_spec, -1 * top_db)  # [-80, 0]
    return log_spec

def mel_frequencies(n_melss=128, fmin=0.0, fmax=11025.0):
    """Compute the center frequencies of mel bands.
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
        min_log_hz = 1000.0                         # beginning of log region
        min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
        logstep = np.log(6.4) / 27.0                # step size for log region
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
        min_log_hz = 1000.0                         # beginning of log region
        min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
        logstep = np.log(6.4) / 27.0                # step size for log region

        log_t = (frequencies >= min_log_hz)
        mels[log_t] = min_log_mel \
                      + np.log(frequencies[log_t] / min_log_hz) / logstep

        return mels

    ''' mel_frequencies body starts '''
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mels = _hz_to_mel(fmin)
    max_mel = _hz_to_mel(fmax)

    mels = np.linspace(min_mels, max_mel, n_melss)

    return _mel_to_hz(mels)


def _dft_frequencies(sr=22050, n_dft=2048):
    '''Alternative implementation of `np.fft.fftfreqs` (said Librosa)
    copied from Librosa

    '''
    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_dft//2),
                       endpoint=True)


def mel(sr, n_dft, n_melss=128, fmin=0.0, fmax=None):
    ''' create a filterbank matrix to combine stft bins into mel-frequency bins
    use Slaney
    Keunwoo: copied from Librosa, librosa.filters.mel
    
    n_melss: numbre of mel bands
    fmin : lowest frequency [Hz]
    fmax : highest frequency [Hz]
        If `None`, use `sr / 2.0`
    '''
    if fmax is None:
        fmax = float(sr) / 2

    # init
    n_melss = int(n_melss)
    weights = np.zeros((n_melss, int(1 + n_dft // 2)))

    # center freqs of each FFT bin
    dftfreqs = _dft_frequencies(sr=sr, n_dft=n_dft)

    # centre freqs of mel bands
    freqs = mel_frequencies(n_melss + 2,
                             fmin=fmin,
                             fmax=fmax)
    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (freqs[2:n_melss+2] - freqs[:n_melss])

    for i in range(n_melss):
        # lower and upper slopes qfor all bins
        lower = (dftfreqs - freqs[i]) / (freqs[i + 1] - freqs[i])
        upper = (freqs[i + 2] - dftfreqs) / (freqs[i + 2] - freqs[i + 1])

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper)) * enorm[i]

    return weights

def get_stft_kernels(n_dft, keras_ver='new'):
    '''Return dft kernels for real/imagnary parts assuming
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

    return dft_real_kernels.astype(dtype), dft_imag_kernels.astype(dtype)


def _hann(M, sym=True):
    '''
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
    return w

