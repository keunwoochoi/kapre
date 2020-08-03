import pytest
import numpy as np
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.backend import image_data_format
import librosa
from kapre.time_frequency import STFT


def _num_frame_valid(nsp_src, nsp_win, len_hop):
    """Computes the number of frames with 'valid' setting"""
    return (nsp_src - (nsp_win - len_hop)) // len_hop


def _num_frame_same(nsp_src, len_hop):
    """Computes the number of frames with 'same' setting"""
    return int(np.ceil(float(nsp_src) / len_hop))


@pytest.mark.parametrize('n_fft', [512, 1000])
@pytest.mark.parametrize('hop_length', [None, 256])
@pytest.mark.parametrize('power', [1.0, 2.0])
@pytest.mark.parametrize('n_ch', [1, 2])
@pytest.mark.parametrize('return_decibel', [True, False])
@pytest.mark.parametrize('data_format', ['default', 'channels_first', 'channels_last'])
def test_spectrogram(n_fft, hop_length, power, n_ch, return_decibel, data_format):
    audio_data = np.load('tests/speech_test_file.npz')['audio_data']
    win_length = n_fft  # test with x2
    hop_length = 128

    # compute with librosa
    S_librosa = librosa.core.stft(audio_data, n_fft=n_fft, hop_length=hop_length,
                                  win_length=win_length, center=False)
    S_librosa = librosa.magphase(S_librosa, power=power)[0].T  # (time, freq)

    # compute with kapre
    stft_model = tensorflow.keras.models.Sequential()
    stft_model.add(
        STFT(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=power,
            pad_end=False,
            return_decibel=return_decibel,
            data_format=data_format,
            input_shape=(1, len(audio_data)),
            name='stft',
        )
    )

    S = stft_model.predict(audio_data.reshape(1, 1, -1))

def test_spectrogram_correctness():
    # todo - compare the value with librosa

if __name__ == '__main__':
    pytest.main([__file__])
