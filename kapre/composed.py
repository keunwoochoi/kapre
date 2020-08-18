from .time_frequency import STFT, Magnitude, Phase, MagnitudeToDecibel, ApplyFilterbank
from . import backend

from tensorflow import keras
from tensorflow.keras import Sequential, Model


def get_stft_mag_phase(
    input_shape=None,
    n_fft=2048,
    win_length=None,
    hop_length=None,
    window_fn=None,
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
        input_shape (None or tuple of integers): input shape of the model if this melspectrogram layer is
            is the first layer of your model (see `keras.model.Sequential()` for more details)
        n_fft (int): number of FFT points in `STFT`
        win_length (int): window length of `STFT`
        hop_length (int): hop length of `STFT`
        window_fn (function or None): windowing function of `STFT`.
            Defaults to `None`, which would follow tf.signal.stft default (hann window at the moment)
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
    """

    waveform_to_stft = STFT(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window_fn=window_fn,
        pad_end=pad_end,
        input_data_format=input_data_format,
        output_data_format=output_data_format,
        input_shape=input_shape,
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

    ch_axis = 1 if output_data_format == 'channels_first' else 3

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
        input_shape (None or tuple of integers): input shape of the model if this melspectrogram layer is
            is the first layer of your model (see `keras.model.Sequential()` for more details)
        n_fft (int): number of FFT points in `STFT`
        win_length (int): window length of `STFT`
        hop_length (int): hop length of `STFT`
        window_fn (function or None): windowing function of `STFT`.
            Defaults to `None`, which would follow tf.signal.stft default (hann window at the moment)
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
        output_data_format (str): the data format of output mel spectrogram.
            `'channels_last'` if you want `(batch, time, frequency, channels)`
            `'channels_first'` if you want `(batch, channels, time, frequency)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())

    """
    waveform_to_stft = STFT(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window_fn=window_fn,
        pad_end=pad_end,
        input_data_format=input_data_format,
        output_data_format=output_data_format,
        input_shape=input_shape,
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
        input_shape (None or tuple of integers): input shape of the model if this melspectrogram layer is
            is the first layer of your model (see `keras.model.Sequential()` for more details)
        n_fft (int): number of FFT points in `STFT`
        win_length (int): window length of `STFT`
        hop_length (int): hop length of `STFT`
        window_fn (function or None): windowing function of `STFT`.
            Defaults to `None`, which would follow tf.signal.stft default (hann window at the moment)
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

    """
    waveform_to_stft = STFT(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window_fn=window_fn,
        pad_end=pad_end,
        input_data_format=input_data_format,
        output_data_format=output_data_format,
        input_shape=input_shape,
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
