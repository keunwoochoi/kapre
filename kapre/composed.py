from .time_frequency import STFT, Magnitude, Phase, MagnitudeToDecibel, ApplyFilterbank
from . import backend

from tensorflow.keras import Sequential


def get_melspectrogram_layer(n_fft=2048,
                             sample_rate=22050,
                             n_mels=128,
                             mel_f_min=0.0,
                             mel_f_max=None,
                             mel_htk=False,
                             mel_norm='slaney',
                             win_length=None,
                             hop_length=None,
                             window_fn=None,
                             pad_end=False,
                             return_decibel=False,
                             input_data_format='default',
                             output_data_format='default',
                             amin=None,
                             dynamic_range=120.0):
    waveform_to_stft = STFT(n_fft=n_fft,
                            win_length=win_length,
                            hop_length=hop_length,
                            window_fn=window_fn,
                            pad_end=pad_end,
                            input_data_format=input_data_format,
                            output_data_format=output_data_format)

    stft_to_stftm = Magnitude()

    _mel_filterbank = backend.filterbank_mel(sample_rate=sample_rate,
                                             n_freq=n_fft // 2 + 1,
                                             n_mels=n_mels,
                                             f_min=mel_f_min,
                                             f_max=mel_f_max,
                                             htk=mel_htk,
                                             norm=mel_norm)
    stftm_to_melgram = ApplyFilterbank(filterbank=_mel_filterbank,
                                       data_format=output_data_format)

    layers = [waveform_to_stft, stft_to_stftm, stftm_to_melgram]
    if return_decibel:
        mag_to_decibel = MagnitudeToDecibel(amin=amin, dynamic_range=dynamic_range)
        layers.append(mag_to_decibel)

    return Sequential(layers)


def get_log_frequency_spectrogram_layer(n_fft=2048,
                                        sample_rate=22050,
                                        log_n_bins=128,
                                        log_f_min=None,
                                        log_bins_per_octave=12,
                                        log_spread=0.125,
                                        win_length=None,
                                        hop_length=None,
                                        window_fn=None,
                                        pad_end=False,
                                        return_decibel=False,
                                        input_data_format='default',
                                        output_data_format='default',
                                        amin=None,
                                        dynamic_range=120.0):
    waveform_to_stft = STFT(n_fft=n_fft,
                            win_length=win_length,
                            hop_length=hop_length,
                            window_fn=window_fn,
                            pad_end=pad_end,
                            input_data_format=input_data_format,
                            output_data_format=output_data_format)

    stft_to_stftm = Magnitude()

    _log_filterbank = backend.filterbank_log(sample_rate=sample_rate,
                                             n_freq=n_fft // 2 + 1,
                                             n_bins=log_n_bins,
                                             bins_per_octave=log_bins_per_octave,
                                             f_min=log_f_min,
                                             spread=log_spread)
    stftm_to_loggram = ApplyFilterbank(filterbank=_log_filterbank,
                                       data_format=output_data_format)

    layers = [waveform_to_stft, stft_to_stftm, stftm_to_loggram]
    
    if return_decibel:
        mag_to_decibel = MagnitudeToDecibel(amin=amin, dynamic_range=dynamic_range)
        layers.append(mag_to_decibel)

    return Sequential(layers)
