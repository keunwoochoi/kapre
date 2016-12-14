''''''
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
import scipy.signal
from keras.models import Sequential
from keras.layers.convolutional import Convolution1D
from keras.layers import Input, Lambda, merge, Permute, Reshape
from keras.models import Model
from keras import backend as K


def Spectrogram(n_dft, input_shape, trainable=False, n_hop=None, 
                border_mode='same', logamplitude=True):
    '''A keras model for Spectrogram using STFT

    # Parameters
    
        n_dft : int > 0 and power of 2 [scalar]
            number of dft components.

        input_shape : tuple (length=2),
            Input shape of raw audio input.
            It should (num_audio_samples, n_ch), e.g. (441000, 1), (16000, 2)

        trainable : boolean
            If it is `True`, the STFT kernels (=weights of two 1d conv layer)
            is set as `trainable`, therefore they are initiated with STFT 
            kernels but then updated.

        n_hop : int > 0 [scalar]
            number of audio samples between successive frames.
    
        border_mode : 'valid' or 'same'.
            if 'valid' the edges of input signal are ignored.

        logamplitude : boolean
            whether logamplitude to stft or not

    # Returns

        A keras model that has output shape of 
            (None, n_ch, n_freq, n_frame) (if `img_dim_ordering() == 'th'`) or
            (None, n_freq, n_frame, n_ch) (if `img_dim_ordering() == 'tf'`).

    '''
    model = get_spectrogram_model(n_dft, input_shape=input_shape,
                                    trainable=trainable,
                                    n_hop=n_hop, 
                                    border_mode=border_mode,
                                    logamplitude=logamplitude)

    model.trainable = trainable
    return model



def Melspectrogram(n_dft, input_shape, trainable, n_hop=None, 
                   border_mode='same', logamplitude=True, sr=22050, 
                   n_mels=128, fmin=0.0, fmax=None, name='melgram'):
    '''Return a Mel-spectrogram keras layer

    # Parameters

        n_dft : int > 0 and power of 2 [scalar]
            number of dft components.

        input_shape : tuple (length=2),
            Input shape of raw audio input.
            It should (num_audio_samples, 1), e.g. (441000, 1)

        trainable : boolean
            If it is `True`, the STFT kernels (=weights of two 1d conv layer)
            AND hz->mel filter banks are set as `trainable`, 
            therefore they are updated. 
        
        n_hop : int > 0 [scalar]
            number of audio samples between successive frames.
            
        border_mode : 'valid' or 'same'.
            if 'valid' the edges of input signal are ignored.

        logamplitude : boolean
            whether logamplitude to stft or not

        sr : int > 0 [scalar]
            sampling rate (used to compute mel-frequency filterbanks)

        n_mels : int > 0 [scalar]
            number of mel-bins

        fmin : float > 0 [scalar]
            minimum frequency of mel-filterbanks

        fmax : float > fmin [scalar]
            maximum frequency of mel-filterbanks

        name : string
            name of the model

    # Returns
        A Keras model that compute mel-spectrogram.
        The output shape follows general 2d-representations,
        i.e., (None, n_ch, height, width) for `theano` or etc.
    '''
    if input_shape is None:
        raise RuntimeError('specify input shape')

    Melgram = Sequential()
    # Prepare STFT.
    stft_model = get_spectrogram_model(n_dft, 
                                        n_hop=n_hop, 
                                        border_mode=border_mode, 
                                        input_shape=input_shape,
                                        logamplitude=False) 
    # output: 2d shape, either (None, 1, freq, time) or..
    stft_model.trainable = trainable
    Melgram.add(stft_model)

    # build a Mel filter
    mel_basis = _mel(sr, n_dft, n_mels, fmin, fmax)  # (128, 1025) (mel_bin, n_freq)
    mel_basis = np.fliplr(mel_basis)  # to make it from low-f to high-freq
    n_freq = mel_basis.shape[1]

    if K.image_dim_ordering() == 'th':
        mel_basis = mel_basis[:, np.newaxis, :, np.newaxis] 
        # print('th', mel_basis.shape)
    else:
        mel_basis = np.transpose(mel_basis, (1, 0))
        mel_basis = mel_basis[:, np.newaxis, np.newaxis, :] 
        # print('tf', mel_basis.shape)
    
    stft2mel = Convolution2D(n_mels, n_freq, 1, border_mode='valid', bias=False,
                            name='stft2mel', weights=[mel_basis])
    stft2mel.trainable = trainable

    Melgram.add(stft2mel)  # output: (None, 128, 1, 375) if theano.
    if logamplitude:
        Melgram.add(Logam_layer())
    # i.e. 128ch == 128 mel-bin, for 375 time-step, therefore,
    if K.image_dim_ordering() == 'th':
        Melgram.add(Permute((2, 1, 3), name='ch_freq_time'))
    else:
        Melgram.add(Permute((3, 2, 1), name='ch_freq_time'))
    # output dot product of them
    return Melgram


def _get_stft_kernels(n_dft, keras_ver='new'):
    '''Return dft kernels for real/imagnary parts assuming
        the input signal is real.
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

    # prepare DFT filters
    timesteps = range(n_dft)
    w_ks = [(2 * np.pi * k) / float(n_dft) for k in xrange(n_dft)]
    dft_real_kernels = np.array([[np.cos(w_k * n) for n in timesteps]
                                  for w_k in w_ks])
    dft_imag_kernels = np.array([[np.sin(w_k * n) for n in timesteps]
                                  for w_k in w_ks])

    # windowing DFT filters
    dft_window = scipy.signal.hann(n_dft, sym=False)
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

    return dft_real_kernels, dft_imag_kernels


def Logam_layer(name='log_amplitude'):
# TODO: replace it with Utils.AmplitudeToDB    
    '''Return a keras layer for log-amplitude.
    The computation is simplified from librosa.logamplitude by
        not having parameters such as ref_power, amin, tob_db.

    Parameters
    ----------
    name : string
        Name of the logamplitude layer

    Returns
    -------
    a Keras layer : Keras's Lambda layer for log-amplitude-ing.
    '''
    def logam(x):
        log_spec = 10 * K.log(K.maximum(x, 1e-10))/K.log(10)
        log_spec = log_spec - K.max(log_spec)  # [-?, 0]
        log_spec = K.maximum(log_spec, -80.0)  # [-80, 0]
        return log_spec

    def logam_shape(shapes):
        '''shapes: shape of input(s) of the layer'''
        # print('output shape of logam:', shapes)
        return shapes

    return Lambda(lambda x: logam(x), name=name,
        output_shape=logam_shape)


def get_spectrogram_model(n_dft, input_shape, trainable=False, 
                            n_hop=None, border_mode='same', 
                            logamplitude=True):
    '''Returns two tensors, x as input, stft_magnitude as result.
        x(input) and STFT_magnitude(tensor) (#freq, #time shape)

    It assumes mono input.
    
    These tensors can be use to build a Keras model 
        using Functional API, 
        `e.g., model = keras.models.Model(x, STFT_magnitude)`
        to build a model that does STFT.
    
    It uses two `Convolution1D` to compute real/imaginary parts of
        STFT and sum(real**2, imag**2). 

    Parameters
    ----------
    n_dft : int > 0 and power of 2 [scalar]
        number of dft components.

    input_shape : tuple (length=2),
        Input shape of raw audio input.
        It should (num_audio_samples, 1), e.g. (441000, 1)

    trainable : boolean
        If it is `True`, the STFT kernels (=weights of two 1d conv layer)
        is set as `trainable`, therefore they are initiated with STFT 
        kernels but then updated. 

    n_hop : int > 0 [scalar]
        number of samples between successive frames.
    
    border_mode : 'valid' or 'same'.
        if 'valid' the edges of input signal are ignored.

    logamplitude : boolean
        whether logamplitude to stft or not


    this is then used in Keras - Functional model API
    STFT_real and STFT_imag is set as non_trainable

    Returns
    -------
    x : input tensor

    STFT_magnitude : STFT magnitude, either in shape:
        (None, 1, n_freq, n_frame) or (None, n_freq, n_frame, 1)
    '''

    assert trainable in (True, False)

    if n_hop is None:
        n_hop = n_dft / 2

    n_channel = input_shape[1]
    # get DFT kernels  
    dft_real_kernels, dft_imag_kernels = _get_stft_kernels(n_dft)
    nb_filter = n_dft / 2 + 1

    # layers - one for the real, one for the imaginary
    x = Input(shape=input_shape, name='audio_input', dtype='float32')

    STFT_real = Convolution1D(nb_filter, n_dft,
                              subsample_length=n_hop,
                              border_mode=border_mode,
                              weights=[dft_real_kernels],
                              bias=False,
                              name='dft_real',
                              input_shape=input_shape)(x)

    STFT_imag = Convolution1D(nb_filter, n_dft,
                              subsample_length=n_hop,
                              border_mode=border_mode,
                              weights=[dft_imag_kernels],
                              bias=False,
                              name='dft_imag',
                              input_shape=input_shape)(x)
    
    STFT_real.trainable = trainable
    STFT_imag.trainable = trainable
    
    STFT_real = Lambda(lambda x: x ** 2, name='real_pow')(STFT_real)
    STFT_imag = Lambda(lambda x: x ** 2, name='imag_pow')(STFT_imag)

    STFT_magnitude = merge([STFT_real, STFT_imag], mode='sum', name='sum')

    if logamplitude:
        STFT_magnitude = Logam_layer()(STFT_magnitude)
    
    STFT_magnitude = Permute((2, 1))(STFT_magnitude)  # (sample, freq, time)
    model_conv1d = Model(input=x, output=STFT_magnitude, name='stft_conv1d')
    model_stft = Sequential(name='stft_model')
    model_stft.add(model_conv1d)
    

    if K.image_dim_ordering() == 'th':
        model_stft.add(Reshape((1, ) + model_conv1d.output_shape[1:]))
    else:
        model_stft.add(Reshape(model_conv1d.output_shape[1:] + (1, )))

    return model_stft


def _mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0):
    """Compute the center frequencies of mel bands.
    `htk` is removed.
    copied from Librosa
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
        copied from Librosa
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
    min_mel = _hz_to_mel(fmin)
    max_mel = _hz_to_mel(fmax)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return _mel_to_hz(mels)


def _dft_frequencies(sr=22050, n_dft=2048):
    '''Alternative implementation of `np.fft.fftfreqs` (said Librosa)
    copied from Librosa

    '''
    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_dft//2),
                       endpoint=True)


def _mel(sr, n_dft, n_mels=128, fmin=0.0, fmax=None):
    ''' create a filterbank matrix to combine stft bins into mel-frequency bins
    use Slaney
    copied from Librosa, librosa.filters.mel
    
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
    freqs = _mel_frequencies(n_mels + 2,
                             fmin=fmin,
                             fmax=fmax)
    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (freqs[2:n_mels+2] - freqs[:n_mels])

    for i in range(n_mels):
        # lower and upper slopes qfor all bins
        lower = (dftfreqs - freqs[i]) / (freqs[i + 1] - freqs[i])
        upper = (freqs[i + 2] - dftfreqs) / (freqs[i + 2] - freqs[i + 1])

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper)) * enorm[i]

    return weights
