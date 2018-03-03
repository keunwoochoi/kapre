import pytest
import numpy as np
import keras
import kapre
import pdb
import librosa
from kapre.time_frequency import Spectrogram


def test_spectrogram():
    src = np.random.uniform(-1., 1., 8000)
    S_ref = librosa.stft(src, n_fft=512, hop_length=256)

    model = keras.models.Sequential()
    model.add(Spectrogram(n_dft=512, n_hop=256, padding='valid',
                          power_spectrogram=1.0, return_decibel_spectrogram=False,
                          image_data_format='default',
                          input_shape=(1, 8000)))
    stft_kapre = model.predict(src[np.newaxis, np.newaxis, :])
    if keras.backend.image_data_format() == 'channels_last':
        S_kapre = stft_kapre[0, :, :, 0]
    else:
        S_kapre = stft_kapre[0, 0, :, :]

    S_ref = S_ref[:, 1:-1]
    S_kapre = S_kapre[:, 1:-1]
    assert np.allclose(S_kapre, np.abs(S_ref), atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
