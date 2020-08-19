"""Layers that are composed using time-frequency layers.

This module provides more complicated layers using layers and operations in Kapre.

"""
from .time_frequency import STFT, InverseSTFT, Magnitude, Phase, MagnitudeToDecibel, ApplyFilterbank
from . import backend

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from .backend import CH_FIRST_STR, CH_LAST_STR, CH_DEFAULT_STR


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

    Note:
        Imagine `x` --> `STFT` --> `InverseSTFT` --> `y`.
        The length of `x` will be longer than `y` due to the padding at the beginning and the end.
        To compare them, you would need to trim `y` along time axis.

        The formula: if `trim_begin = win_length - hop_length` and `len_signal` is length of `x`,
        `y_trimmed = y[trim_begin: trim_begin + len_signal, :]` (in the case of `channels_last`).


    Args:
        stft_input_shape (tuple): Input shape of single waveform.
            Must specify this if the returned stft layer is going to be used as first layer of a Sequential model.
        istft_input_shape (tuple): Input shape of single STFT.
            Must specify this if the returned istft layer is going to be used as first layer of a Sequential model.
        n_fft (`int`): Number of FFTs. Defaults to `2048`
        win_length (`int` or `None`): Window length in sample. Defaults to `n_fft`.
        hop_length (`int` or `None`): Hop length in sample between analysis windows. Defaults to `n_fft // 4` following Librosa.
        forward_window_fn (function or `None`): A function that returns a 1D tensor window. Defaults to `tf.signal.hann_window`.
        waveform_data_format (`str`): The audio data format of waveform batch.
            `'channels_last'` if it's `(batch, time, channels)`
            `'channels_first'` if it's `(batch, channels, time)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())
        stft_data_format (`str`): The data format of STFT.
            `'channels_last'` if you want `(batch, time, frequency, channels)`
            `'channels_first'` if you want `(batch, channels, time, frequency)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())
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
        input_shape (`None` or tuple of integers): input shape of the stft layer.
            Because this mag_phase is based on keras.Functional model, it is required to specify the input shape.
            E.g., (44100, 2) for 44100-sample stereo audio with `input_data_format=='channels_last'`.
        n_fft (`int`): number of FFT points in `STFT`
        win_length (`int`): window length of `STFT`
        hop_length (`int`): hop length of `STFT`
        window_fn (function or `None`): windowing function of `STFT`.
            Defaults to `None`, which would follow tf.signal.stft default (hann window at the moment)
        pad_begin(`bool`): Whether to pad with zeros along time axis (legnth: win_length - hop_length). Defaults to `False`.
        pad_end (`bool`): whether to pad the input signal at the end in `STFT`.
        return_decibel (`bool`): whether to apply decibel scaling at the end
        db_amin (`float`): noise floor of decibel scaling input. See `MagnitudeToDecibel` for more details.
        db_ref_value (`float`): reference value of decibel scaling. See `MagnitudeToDecibel` for more details.
        db_dynamic_range (`float`): dynamic range of the decibel scaling result.
        input_data_format (`str`): the audio data format of input waveform batch.
            `'channels_last'` if it's `(batch, time, channels)`
            `'channels_first'` if it's `(batch, channels, time)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())
        output_data_format (`str`): the data format of output mel spectrogram.
            `'channels_last'` if you want `(batch, time, frequency, channels)`
            `'channels_first'` if you want `(batch, channels, time, frequency)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())
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

    ch_axis = 1 if output_data_format == CH_FIRST_STR else 3

    concat_layer = keras.layers.Concatenate(axis=ch_axis)

    stfts_mag_phase = concat_layer([mag_stfts, phase_stfts])

    model = Model(inputs=waveforms, outputs=stfts_mag_phase)
    return model


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
    """A function that retunrs a melspectrogram layer, which is a Sequential model consists of
    `STFT`, `Magnitude`, `ApplyFilterbank(_mel_filterbank)`, and optionally `MagnitudeToDecibel`.

    Args:
        input_shape (`None` or tuple of integers): input shape of the model if this melspectrogram layer is
            is the first layer of your model (see `keras.model.Sequential()` for more details)
        n_fft (`int`): number of FFT points in `STFT`
        win_length (`int`): window length of `STFT`
        hop_length (`int`): hop length of `STFT`
        window_fn (function or `None`): windowing function of `STFT`.
            Defaults to `None`, which would follow tf.signal.stft default (hann window at the moment)
        pad_begin(`bool`): Whether to pad with zeros along time axis (legnth: win_length - hop_length). Defaults to `False`.
        pad_end (`bool`): whether to pad the input signal at the end in `STFT`.
        sample_rate (`int`): sample rate of the input audio
        n_mels (`int`): number of mel bins in the mel filterbank
        mel_f_min (`float`): lowest frequency of the mel filterbank
        mel_f_max (`float`): highest frequency of the mel filterbank
        mel_htk (`bool`): whether to follow the htk mel filterbank fomula or not
        mel_norm ('slaney' or int): normalization policy of the mel filterbank triangles
        return_decibel (`bool`): whether to apply decibel scaling at the end
        db_amin (`float`): noise floor of decibel scaling input. See `MagnitudeToDecibel` for more details.
        db_ref_value (`float`): reference value of decibel scaling. See `MagnitudeToDecibel` for more details.
        db_dynamic_range (`float`): dynamic range of the decibel scaling result.
        input_data_format (`str`): the audio data format of input waveform batch.
            `'channels_last'` if it's `(batch, time, channels)`
            `'channels_first'` if it's `(batch, channels, time)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())
        output_data_format (`str`): the data format of output melspectrogram.
            `'channels_last'` if you want `(batch, time, frequency, channels)`
            `'channels_first'` if you want `(batch, channels, time, frequency)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())

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
    """A function that retunrs a log-frequency STFT layer, which is a Sequential model consists of
    `STFT`, `Magnitude`, `ApplyFilterbank(_log_filterbank)`, and optionally `MagnitudeToDecibel`.

    Args:
        input_shape (`None` or tuple of integers): input shape of the model if this melspectrogram layer is
            is the first layer of your model (see `keras.model.Sequential()` for more details)
        n_fft (`int`): number of FFT points in `STFT`
        win_length (`int`): window length of `STFT`
        hop_length (`int`): hop length of `STFT`
        window_fn (function or `None`): windowing function of `STFT`.
            Defaults to `None`, which would follow tf.signal.stft default (hann window at the moment)
        pad_begin(`bool`): Whether to pad with zeros along time axis (legnth: win_length - hop_length). Defaults to `False`.
        pad_end (`bool`): whether to pad the input signal at the end in `STFT`.
        sample_rate (`int`): sample rate of the input audio
        log_n_bins (`int`): number of the bins in the log-frequency filterbank
        log_f_min (`float`): lowest frequency of the filterbank
        log_bins_per_octave (`int`): number of bins in each octave in the filterbank
        log_spread (`float`): spread constant (Q value) in the log filterbank.
        return_decibel (`bool`): whether to apply decibel scaling at the end
        db_amin (`float`): noise floor of decibel scaling input. See `MagnitudeToDecibel` for more details.
        db_ref_value (`float`): reference value of decibel scaling. See `MagnitudeToDecibel` for more details.
        db_dynamic_range (`float`): dynamic range of the decibel scaling result.
        input_data_format (`str`): the audio data format of input waveform batch.
            `'channels_last'` if it's `(batch, time, channels)`
            `'channels_first'` if it's `(batch, channels, time)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())
        output_data_format (`str`): the data format of output mel spectrogram.
            `'channels_last'` if you want `(batch, time, frequency, channels)`
            `'channels_first'` if you want `(batch, channels, time, frequency)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())

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
