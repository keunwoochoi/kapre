import pytest
import librosa
import numpy as np
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.backend import image_data_format

import kapre
from kapre.delta import ComputeDeltas


def test_deltas():
    def _test_correctness():
        """ Tests correctness
        """
        audio_data = np.load('tests/speech_test_file.npz')['audio_data']
        sr = 44100

        hop_length = 128
        n_fft = 1024
        n_mels = 80

        # compute with librosa
        S = librosa.feature.melspectrogram(
            audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

        DS_librosa = librosa.feature.delta(S, width=5, mode='constant')

        # compute with kapre
        mels_model = tensorflow.keras.models.Sequential()
        mels_model.add(ComputeDeltas(win_length=5, mode='CONSTANT'))

        if image_data_format() == 'channels_last':
            S_input = S.reshape(1, -1, S.shape[-1], 1)
        else:
            S_input = S.reshape(1, 1, -1, S.shape[-1])

        DS = mels_model.predict(S_input)

        if image_data_format() == 'channels_last':
            DS = DS[0, :, :, 0]
        else:
            DS = DS[0, 0]

        DB_DS = librosa.power_to_db(DS, ref=np.max)

        np.testing.assert_allclose(DS_librosa, DS, rtol=1e-02)

    K.set_image_data_format("channels_first")
    _test_correctness()

    K.set_image_data_format("channels_last")
    _test_correctness()
