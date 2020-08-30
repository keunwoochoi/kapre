"""Functions that returns high-level layers that are composed using other Kapre layers.

"""
from .time_frequency import STFT, InverseSTFT, Magnitude, Phase, MagnitudeToDecibel, ApplyFilterbank
from . import backend

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from .backend import _CH_FIRST_STR, _CH_LAST_STR, _CH_DEFAULT_STR


def get_stft_magnitude_layer(
    input_shape=None,
    n_fft=2048,
    win_length=None,
    hop_length=None,
    window_fn=None,
    pad_begin=False,
    pad_end=False,
    return_decibel=False,
    db_amin=1e-5,
    db_ref_value=1.0,
    db_dynamic_range=80.0,
    input_data_format='default',
    output_data_format='default',
):
    """A function that retunrs a stft magnitude layer.
    The layer is a `keras.Sequential` model consists of `STFT`, `Magnitude`, and optionally `MagnitudeToDecibel`.

    Args:
        input_shape (None or tuple of integers): input shape of the model. Necessary only if this melspectrogram layer is
            is the first layer of your model (see `keras.model.Sequential()` for more details)
        n_fft (int): number of FFT points in `STFT`
        win_length (int): window length of `STFT`
        hop_length (int): hop length of `STFT`
        window_fn (function or `None`): windowing function of `STFT`.
            Defaults to `None`, which would follow tf.signal.stft default (hann window at the moment)
        pad_begin (bool): Whether to pad with zeros along time axis (legnth: win_length - hop_length). Defaults to `False`.
        pad_end (bool): whether to pad the input signal at the end in `STFT`.
        return_decibel (bool): whether to apply decibel scaling at the end
        db_amin (float): noise floor of decibel scaling input. See `MagnitudeToDecibel` for more details.
        db_ref_value (float): reference value of decibel scaling. See `MagnitudeToDecibel` for more details.
        db_dynamic_range (float): dynamic range of the decibel scaling result.
        input_data_format (str): the audio data format of input waveform batch.
            `'channels_last'` if it's `(batch, time, channels)`
            `'channels_first'` if it's `(batch, channels, time)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())
        output_data_format (str): the data format of output melspectrogram.
            `'channels_last'` if you want `(batch, time, frequency, channels)`
            `'channels_first'` if you want `(batch, channels, time, frequency)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())

    Note:
        STFT magnitude represents a linear-frequency spectrum of audio signal and probably the most popular choice
        when it comes to audio analysis in general. By using magnitude, this layer discard the phase information,
        which is generally known to be irrelevant to human auditory perception.

    Note:
        For audio analysis (when the output is tag/label/etc), we'd like to recommend to set `return_decibel=True`.
        Decibel scaling is perceptually plausible and numerically stable
        (related paper: `A Comparison of Audio Signal Preprocessing Methods for Deep Neural Networks on Music Tagging <https://arxiv.org/abs/1709.01922>`_)
        Many music, speech, and audio applications have used this log-magnitude STFT, e.g.,
        `Learning to Pinpoint Singing Voice from Weakly Labeled Examples <https://wp.nyu.edu/ismir2016/wp-content/uploads/sites/2294/2016/07/315_Paper.pdf>`_,
        `Joint Beat and Downbeat Tracking with Recurrent Neural Networks <https://archives.ismir.net/ismir2016/paper/000186.pdf>`_,
        and many more.

        For audio processing (when the output is audio signal), it might be better to use STFT as it is (`return_decibel=False`).
        Example: `Singing voice separation with deep U-Net convolutional networks <https://openaccess.city.ac.uk/id/eprint/19289/>`_.
        This is because decibel scaling is has some clipping at the noise floor which is irreversible.
        One may use `log(1+X)` instead of `log(X)` to avoid the clipping but it is not included in Kapre at the moment.

    Example:
        ::

            input_shape = (2048, 1)  # mono signal, audio is channels_last
            stft_mag = get_stft_magnitude_layer(input_shape=input_shape, n_fft=1024, return_decibel=True,
                input_data_format='channels_last', output_data_format='channels_first')
            model = Sequential()
            model.add(stft_mag)
            # now the shape is (batch, ch=1, n_frame=3, n_freq=513) because output_data_format is 'channels_first'
            # and the dtype is float

    """
    backend.validate_data_format_str(input_data_format)
    backend.validate_data_format_str(output_data_format)

    stft_kwargs = {}
    if input_shape is not None:
        stft_kwargs['input_shape'] = input_shape

    waveform_to_stft = STFT(
        **stft_kwargs,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window_fn=window_fn,
        pad_begin=pad_begin,
        pad_end=pad_end,
        input_data_format=input_data_format,
        output_data_format=output_data_format,
    )

    stft_to_stftm = Magnitude()

    layers = [waveform_to_stft, stft_to_stftm]
    if return_decibel:
        mag_to_decibel = MagnitudeToDecibel(
            ref_value=db_ref_value, amin=db_amin, dynamic_range=db_dynamic_range
        )
        layers.append(mag_to_decibel)

    return Sequential(layers)


def get_melspectrogram_layer(
    input_shape=None,
    n_fft=2048,
    win_length=None,
    hop_length=None,
    window_fn=None,
    pad_begin=False,
    pad_end=False,
    sample_rate=22050,
    n_mels=128,
    mel_f_min=0.0,
    mel_f_max=None,
    mel_htk=False,
    mel_norm='slaney',
    return_decibel=False,
    db_amin=1e-5,
    db_ref_value=1.0,
    db_dynamic_range=80.0,
    input_data_format='default',
    output_data_format='default',
):
    """A function that retunrs a melspectrogram layer, which is a `keras.Sequential` model consists of
    `STFT`, `Magnitude`, `ApplyFilterbank(_mel_filterbank)`, and optionally `MagnitudeToDecibel`.

    Args:
        input_shape (None or tuple of integers): input shape of the model. Necessary only if this melspectrogram layer is
            is the first layer of your model (see `keras.model.Sequential()` for more details)
        n_fft (int): number of FFT points in `STFT`
        win_length (int): window length of `STFT`
        hop_length (int): hop length of `STFT`
        window_fn (function or `None`): windowing function of `STFT`.
            Defaults to `None`, which would follow tf.signal.stft default (hann window at the moment)
        pad_begin (bool): Whether to pad with zeros along time axis (legnth: win_length - hop_length). Defaults to `False`.
        pad_end (bool): whether to pad the input signal at the end in `STFT`.
        sample_rate (int): sample rate of the input audio
        n_mels (int): number of mel bins in the mel filterbank
        mel_f_min (float): lowest frequency of the mel filterbank
        mel_f_max (float): highest frequency of the mel filterbank
        mel_htk (bool): whether to follow the htk mel filterbank fomula or not
        mel_norm ('slaney' or int): normalization policy of the mel filterbank triangles
        return_decibel (bool): whether to apply decibel scaling at the end
        db_amin (float): noise floor of decibel scaling input. See `MagnitudeToDecibel` for more details.
        db_ref_value (float): reference value of decibel scaling. See `MagnitudeToDecibel` for more details.
        db_dynamic_range (float): dynamic range of the decibel scaling result.
        input_data_format (str): the audio data format of input waveform batch.
            `'channels_last'` if it's `(batch, time, channels)`
            `'channels_first'` if it's `(batch, channels, time)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())
        output_data_format (str): the data format of output melspectrogram.
            `'channels_last'` if you want `(batch, time, frequency, channels)`
            `'channels_first'` if you want `(batch, channels, time, frequency)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())

    Note:
        Melspectrogram is originally developed for speech applications and has been *very* widely used for audio signal
        analysis including music information retrieval. As its mel-axis is a non-linear compression of (linear)
        frequency axis, a melspectrogram can be an efficient choice as an input of a machine learning model.
        We recommend to set `return_decibel=True`.

        **References**:
        `Automatic tagging using deep convolutional neural networks <https://arxiv.org/abs/1606.00298>`_,
        `Deep content-based music recommendation <http://papers.nips.cc/paper/5004-deep-content-based-music-recommen>`_,
        `CNN Architectures for Large-Scale Audio Classification <https://arxiv.org/abs/1609.09430>`_,
        `Multi-label vs. combined single-label sound event detection with deep neural networks <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.711.74&rep=rep1&type=pdf>`_,
        `Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification <https://arxiv.org/pdf/1608.04363.pdf>`_,
        and way too many speech applications.

    Example:
        ::

            input_shape = (2, 2048)  # stereo signal, audio is channels_first
            melgram = get_melspectrogram_layer(input_shape=input_shape, n_fft=1024, return_decibel=True,
                n_mels=96, input_data_format='channels_first', output_data_format='channels_last')
            model = Sequential()
            model.add(melgram)
            # now the shape is (batch, n_frame=3, n_mels=96, n_ch=2) because output_data_format is 'channels_last'
            # and the dtype is float

    """
    backend.validate_data_format_str(input_data_format)
    backend.validate_data_format_str(output_data_format)

    stft_kwargs = {}
    if input_shape is not None:
        stft_kwargs['input_shape'] = input_shape

    waveform_to_stft = STFT(
        **stft_kwargs,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window_fn=window_fn,
        pad_begin=pad_begin,
        pad_end=pad_end,
        input_data_format=input_data_format,
        output_data_format=output_data_format,
    )

    stft_to_stftm = Magnitude()

    kwargs = {
        'sample_rate': sample_rate,
        'n_freq': n_fft // 2 + 1,
        'n_mels': n_mels,
        'f_min': mel_f_min,
        'f_max': mel_f_max,
        'htk': mel_htk,
        'norm': mel_norm,
    }
    stftm_to_melgram = ApplyFilterbank(
        type='mel', filterbank_kwargs=kwargs, data_format=output_data_format
    )

    layers = [waveform_to_stft, stft_to_stftm, stftm_to_melgram]
    if return_decibel:
        mag_to_decibel = MagnitudeToDecibel(
            ref_value=db_ref_value, amin=db_amin, dynamic_range=db_dynamic_range
        )
        layers.append(mag_to_decibel)

    return Sequential(layers)


def get_log_frequency_spectrogram_layer(
    input_shape=None,
    n_fft=2048,
    win_length=None,
    hop_length=None,
    window_fn=None,
    pad_begin=False,
    pad_end=False,
    sample_rate=22050,
    log_n_bins=84,
    log_f_min=None,
    log_bins_per_octave=12,
    log_spread=0.125,
    return_decibel=False,
    db_amin=1e-5,
    db_ref_value=1.0,
    db_dynamic_range=80.0,
    input_data_format='default',
    output_data_format='default',
):
    """A function that retunrs a log-frequency STFT layer, which is a `keras.Sequential` model consists of
    `STFT`, `Magnitude`, `ApplyFilterbank(_log_filterbank)`, and optionally `MagnitudeToDecibel`.

    Args:
        input_shape (None or tuple of integers): input shape of the model if this melspectrogram layer is
            is the first layer of your model (see `keras.model.Sequential()` for more details)
        n_fft (int): number of FFT points in `STFT`
        win_length (int): window length of `STFT`
        hop_length (int): hop length of `STFT`
        window_fn (function or `None`): windowing function of `STFT`.
            Defaults to `None`, which would follow tf.signal.stft default (hann window at the moment)
        pad_begin(bool): Whether to pad with zeros along time axis (legnth: win_length - hop_length). Defaults to `False`.
        pad_end (bool): whether to pad the input signal at the end in `STFT`.
        sample_rate (int): sample rate of the input audio
        log_n_bins (int): number of the bins in the log-frequency filterbank
        log_f_min (float): lowest frequency of the filterbank
        log_bins_per_octave (int): number of bins in each octave in the filterbank
        log_spread (float): spread constant (Q value) in the log filterbank.
        return_decibel (bool): whether to apply decibel scaling at the end
        db_amin (float): noise floor of decibel scaling input. See `MagnitudeToDecibel` for more details.
        db_ref_value (float): reference value of decibel scaling. See `MagnitudeToDecibel` for more details.
        db_dynamic_range (float): dynamic range of the decibel scaling result.
        input_data_format (str): the audio data format of input waveform batch.
            `'channels_last'` if it's `(batch, time, channels)`
            `'channels_first'` if it's `(batch, channels, time)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())
        output_data_format (str): the data format of output mel spectrogram.
            `'channels_last'` if you want `(batch, time, frequency, channels)`
            `'channels_first'` if you want `(batch, channels, time, frequency)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())

    Note:
        Log-frequency spectrogram is similar to melspectrogram but its frequency axis is perfectly linear to octave scale.
        For some pitch-related applications, a log-frequency spectrogram can be a good choice.

    Example:
        ::

            input_shape = (2048, 2)  # stereo signal, audio is channels_last
            logfreq_stft_mag = get_log_frequency_spectrogram_layer(
                input_shape=input_shape, n_fft=1024, return_decibel=True,
                log_n_bins=84, input_data_format='channels_last', output_data_format='channels_last')
            model = Sequential()
            model.add(logfreq_stft_mag)
            # now the shape is (batch, n_frame=3, n_bins=84, n_ch=2) because output_data_format is 'channels_last'
            # and the dtype is float

    """
    backend.validate_data_format_str(input_data_format)
    backend.validate_data_format_str(output_data_format)

    stft_kwargs = {}
    if input_shape is not None:
        stft_kwargs['input_shape'] = input_shape

    waveform_to_stft = STFT(
        **stft_kwargs,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window_fn=window_fn,
        pad_begin=pad_begin,
        pad_end=pad_end,
        input_data_format=input_data_format,
        output_data_format=output_data_format,
    )

    stft_to_stftm = Magnitude()

    _log_filterbank = backend.filterbank_log(
        sample_rate=sample_rate,
        n_freq=n_fft // 2 + 1,
        n_bins=log_n_bins,
        bins_per_octave=log_bins_per_octave,
        f_min=log_f_min,
        spread=log_spread,
    )
    kwargs = {
        'sample_rate': sample_rate,
        'n_freq': n_fft // 2 + 1,
        'n_bins': log_n_bins,
        'bins_per_octave': log_bins_per_octave,
        'f_min': log_f_min,
        'spread': log_spread,
    }

    stftm_to_loggram = ApplyFilterbank(
        type='log', filterbank_kwargs=kwargs, data_format=output_data_format
    )

    layers = [waveform_to_stft, stft_to_stftm, stftm_to_loggram]

    if return_decibel:
        mag_to_decibel = MagnitudeToDecibel(
            ref_value=db_ref_value, amin=db_amin, dynamic_range=db_dynamic_range
        )
        layers.append(mag_to_decibel)

    return Sequential(layers)


def get_perfectly_reconstructing_stft_istft(
    stft_input_shape=None,
    istft_input_shape=None,
    n_fft=2048,
    win_length=None,
    hop_length=None,
    forward_window_fn=None,
    waveform_data_format='default',
    stft_data_format='default',
):
    """A function that returns two layers, stft and inverse stft, which would be perfectly reconstructing pair.

    Args:
        stft_input_shape (tuple): Input shape of single waveform.
            Must specify this if the returned stft layer is going to be used as first layer of a Sequential model.
        istft_input_shape (tuple): Input shape of single STFT.
            Must specify this if the returned istft layer is going to be used as first layer of a Sequential model.
        n_fft (int): Number of FFTs. Defaults to `2048`
        win_length (`int` or `None`): Window length in sample. Defaults to `n_fft`.
        hop_length (`int` or `None`): Hop length in sample between analysis windows. Defaults to `n_fft // 4` following librosa.
        forward_window_fn (function or `None`): A function that returns a 1D tensor window. Defaults to `tf.signal.hann_window`.
        waveform_data_format (str): The audio data format of waveform batch.
            `'channels_last'` if it's `(batch, time, channels)`
            `'channels_first'` if it's `(batch, channels, time)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())
        stft_data_format (str): The data format of STFT.
            `'channels_last'` if you want `(batch, time, frequency, channels)`
            `'channels_first'` if you want `(batch, channels, time, frequency)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())

    Note:
        Without a careful setting, `tf.signal.stft` and `tf.signal.istft` is not perfectly reconstructing.

    Note:
        Imagine `x` --> `STFT` --> `InverseSTFT` --> `y`.
        The length of `x` will be longer than `y` due to the padding at the beginning and the end.
        To compare them, you would need to trim `y` along time axis.

        The formula: if `trim_begin = win_length - hop_length` and `len_signal` is length of `x`,
        `y_trimmed = y[trim_begin: trim_begin + len_signal, :]` (in the case of `channels_last`).

    Example:
        ::

            stft_input_shape = (2048, 2)  # stereo and channels_last
            stft_layer, istft_layer = get_perfectly_reconstructing_stft_istft(
                stft_input_shape=stft_input_shape
            )

            unet = get_unet()  input: stft (complex value), output: stft (complex value)

            model = Sequential()
            model.add(stft_layer)  # input is waveform
            model.add(unet)
            model.add(istft_layer)  # output is also waveform

    """
    backend.validate_data_format_str(waveform_data_format)
    backend.validate_data_format_str(stft_data_format)

    if forward_window_fn is None:
        forward_window_fn = tf.signal.hann_window

    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = win_length // 4

    if (win_length / hop_length) % 2 != 0:
        raise RuntimeError(
            'The ratio of win_length and hop_length must be power of 2 to get a '
            'perfectly reconstructing stft-istft pair.'
        )

    backward_window_fn = tf.signal.inverse_stft_window_fn(
        frame_step=int(hop_length), forward_window_fn=forward_window_fn
    )

    stft_kwargs = {}
    if stft_input_shape is not None:
        stft_kwargs['input_shape'] = stft_input_shape

    istft_kwargs = {}
    if istft_input_shape is not None:
        istft_kwargs['input_shape'] = istft_input_shape

    waveform_to_stft = STFT(
        **stft_kwargs,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window_fn=forward_window_fn,
        pad_begin=True,
        pad_end=True,
        input_data_format=waveform_data_format,
        output_data_format=stft_data_format,
    )

    stft_to_waveform = InverseSTFT(
        **istft_kwargs,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window_fn=backward_window_fn,
        input_data_format=stft_data_format,
        output_data_format=waveform_data_format,
    )

    return waveform_to_stft, stft_to_waveform


def get_stft_mag_phase(
    input_shape,
    n_fft=2048,
    win_length=None,
    hop_length=None,
    window_fn=None,
    pad_begin=False,
    pad_end=False,
    return_decibel=False,
    db_amin=1e-5,
    db_ref_value=1.0,
    db_dynamic_range=80.0,
    input_data_format='default',
    output_data_format='default',
):
    """A function that returns magnitude and phase of input audio.

    Args:
        input_shape (None or tuple of integers): input shape of the stft layer.
            Because this mag_phase is based on keras.Functional model, it is required to specify the input shape.
            E.g., (44100, 2) for 44100-sample stereo audio with `input_data_format=='channels_last'`.
        n_fft (int): number of FFT points in `STFT`
        win_length (int): window length of `STFT`
        hop_length (int): hop length of `STFT`
        window_fn (function or `None`): windowing function of `STFT`.
            Defaults to `None`, which would follow tf.signal.stft default (hann window at the moment)
        pad_begin(bool): Whether to pad with zeros along time axis (legnth: win_length - hop_length). Defaults to `False`.
        pad_end (bool): whether to pad the input signal at the end in `STFT`.
        return_decibel (bool): whether to apply decibel scaling at the end
        db_amin (float): noise floor of decibel scaling input. See `MagnitudeToDecibel` for more details.
        db_ref_value (float): reference value of decibel scaling. See `MagnitudeToDecibel` for more details.
        db_dynamic_range (float): dynamic range of the decibel scaling result.
        input_data_format (str): the audio data format of input waveform batch.
            `'channels_last'` if it's `(batch, time, channels)`
            `'channels_first'` if it's `(batch, channels, time)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())
        output_data_format (str): the data format of output mel spectrogram.
            `'channels_last'` if you want `(batch, time, frequency, channels)`
            `'channels_first'` if you want `(batch, channels, time, frequency)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())

    Example:
        ::

            input_shape = (2048, 3)  # stereo and channels_last
            model = Sequential()
            model.add(
                get_stft_mag_phase(input_shape=input_shape, return_decibel=True, n_fft=1024)
            )
            # now output shape is (batch, n_frame=3, freq=513, ch=6). 6 channels = [3 mag ch; 3 phase ch]

    """
    backend.validate_data_format_str(input_data_format)
    backend.validate_data_format_str(output_data_format)

    waveform_to_stft = STFT(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window_fn=window_fn,
        pad_begin=pad_begin,
        pad_end=pad_end,
        input_data_format=input_data_format,
        output_data_format=output_data_format,
    )

    stft_to_stftm = Magnitude()
    stft_to_stftp = Phase()

    waveforms = keras.Input(shape=input_shape)

    stfts = waveform_to_stft(waveforms)
    mag_stfts = stft_to_stftm(stfts)  # magnitude
    phase_stfts = stft_to_stftp(stfts)  # phase

    if return_decibel:
        mag_to_decibel = MagnitudeToDecibel(
            ref_value=db_ref_value, amin=db_amin, dynamic_range=db_dynamic_range
        )
        mag_stfts = mag_to_decibel(mag_stfts)

    ch_axis = 1 if output_data_format == _CH_FIRST_STR else 3

    concat_layer = keras.layers.Concatenate(axis=ch_axis)

    stfts_mag_phase = concat_layer([mag_stfts, phase_stfts])

    model = Model(inputs=waveforms, outputs=stfts_mag_phase)
    return model
