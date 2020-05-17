import tensorflow.keras
import pytest
import numpy as np
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.backend import image_data_format
import librosa
from kapre.utils import AmplitudeToDB, Normalization2D, Delta

TOL = 1e-5


def test_deltas():
    def _test_correctness():
        """ Tests correctness of `Delta` layer w.r.t. librosa.feature.delta
        """
        audio_data = np.load('tests/speech_test_file.npz')['audio_data']
        sr = 44100

        hop_length = 128
        n_fft = 1024
        n_mels = 80

        # compute with librosa
        S = librosa.feature.melspectrogram(
            audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )  # (mel_bin, time)

        DS_librosa = librosa.feature.delta(S, width=5, mode='constant')

        # compute with kapre
        model_delta = tensorflow.keras.models.Sequential()
        model_delta.add(Delta(win_length=5, mode='CONSTANT'))

        if image_data_format() == 'channels_last':
            S_input = S.reshape(1, S.shape[0], S.shape[1], 1)
        else:
            S_input = S.reshape(1, 1, S.shape[0], S.shape[1])

        DS_kapre = model_delta.predict(S_input)

        if image_data_format() == 'channels_last':
            DS_kapre = DS_kapre[0, :, :, 0]
        else:
            DS_kapre = DS_kapre[0, 0]

        np.testing.assert_allclose(DS_librosa, DS_kapre, rtol=1e-02)

    K.set_image_data_format("channels_first")
    _test_correctness()

    K.set_image_data_format("channels_last")
    _test_correctness()


def test_amplitude_to_db():
    """test for AmplitudeToDB layer"""

    # Test for a normal case
    model = tensorflow.keras.models.Sequential()
    model.add(AmplitudeToDB(amin=1e-10, top_db=80.0, input_shape=(6,)))

    x = np.array([0, 1e-5, 1e-3, 1e-2, 1e-1, 1])
    x_db_ref = np.array([-80, -50, -30, -20, -10, 0])
    batch_x_db = model.predict(x[np.newaxis, :])
    assert np.allclose(batch_x_db[0], x_db_ref, atol=TOL)

    # Smaller amin, bigger dynamic range
    model = tensorflow.keras.models.Sequential()
    model.add(AmplitudeToDB(amin=1e-12, top_db=120.0, input_shape=(6,)))
    x = np.array([1e-15, 1e-10, 1e-5, 1e-2, 1e-1, 10])
    x_db_ref = np.array([-120, -110, -60, -30, -20, 0])
    batch_x_db = model.predict(x[np.newaxis, :])
    assert np.allclose(batch_x_db[0], x_db_ref, atol=TOL)

    # TODO: Saving and loading the model


def test_normalization_2d():
    """test for Normalization2D"""
    # TODO: Because the expected behaviour of this layer is somehow confusing for me now


if __name__ == '__main__':
    pytest.main([__file__])
