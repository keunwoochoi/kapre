import pytest
import torch
import librosa
import torchaudio
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
        audio_data = np.load('./speech_test_file.npz')['audio_data']
        sr = 44100

        hop_length = 128
        n_fft = 1024
        n_mels = 80

        # compute with librosa
        S = librosa.feature.melspectrogram(audio_data, sr=sr, n_fft=n_fft,
                                           hop_length=hop_length,
                                           n_mels=n_mels)

        S_DB_librosa = librosa.power_to_db(S, ref=np.max)

        DS_librosa = librosa.feature.delta(S_DB_librosa)
        
        # Compute with torch 
        DS_torch = (torchaudio.transforms.ComputeDeltas()(torch.Tensor(S_DB_librosa).float())).numpy()


        # compute with kapre
        mels_model = tensorflow.keras.models.Sequential()
        mels_model.add(ComputeDeltas())
        
        if image_data_format() == 'channels_last':
            S_input = S_DB_librosa.reshape(1,-1,S_DB_librosa.shape[-1],1)
        else:
            S_input = S_DB_librosa.reshape(1,1,-1,S_DB_librosa.shape[-1])
            
        S = mels_model.predict(S_input)

        if image_data_format() == 'channels_last':
            DS = S[0, :, :, 0]
        else:
            DS = S[0, 0]

        DS_scale_librosa = (np.max(DS_librosa) - np.min(DS_librosa))
        DS_dif_librosa = np.abs(DS - DS_librosa) / DS_scale_librosa

        DS_scale_torch = (np.max(DS_torch) - np.min(DS_torch))
        DS_dif_torch = np.abs(DS - DS_torch) / DS_scale_torch

        assert np.mean(DS_dif_torch) < 0.001
        assert np.mean(DS_dif_librosa) < 0.05
    
    
    K.set_image_data_format("channels_first")
    _test_correctness()

    K.set_image_data_format("channels_last")
    _test_correctness()