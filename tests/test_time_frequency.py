import pytest
import numpy as np
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.backend import image_data_format
import kapre
import pdb
import librosa
from kapre.time_frequency import Spectrogram, Melspectrogram


def _num_frame_valid(nsp_src, nsp_win, len_hop):
    """Computes the number of frames with 'valid' setting"""
    return (nsp_src - (nsp_win - len_hop)) // len_hop


def _num_frame_same(nsp_src, len_hop):
    """Computes the number of frames with 'same' setting"""
    return int(np.ceil(float(nsp_src) / len_hop))


def test_spectrogram():
    def _test_correctness():
        """ Tests correctness
        """
        audio_data, sr = librosa.load("speech_test_file.wav", sr=44100, mono=True)

        hop_length = 128
        n_fft = 1024
        n_mels = 80

        # compute with librosa
        S = librosa.core.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        magnitudes_librosa = librosa.magphase(S, power=2)[0]
        S_DB_librosa = librosa.power_to_db(magnitudes_librosa, ref=np.max)

        # compute with kapre
        stft_model = tensorflow.keras.models.Sequential()
        stft_model.add(Spectrogram(n_dft=n_fft, n_hop=hop_length, input_shape=(1, len(audio_data)),
                                   power_spectrogram=2.0, return_decibel_spectrogram=False,
                                   trainable_kernel=False, name='stft'))

        S = stft_model.predict(audio_data.reshape(1, 1, -1))
        if image_data_format() == 'channels_last':
            S = S[0, :, :, 0]
        else:
            S = S[0, 0]
        magnitudes_kapre = librosa.magphase(S, power=1)[0]
        S_DB_kapre = librosa.power_to_db(magnitudes_kapre, ref=np.max)

        DB_scale = (np.max(S_DB_librosa) - np.min(S_DB_librosa))
        S_DB_dif = np.abs(S_DB_kapre - S_DB_librosa) / DB_scale
        assert np.mean(S_DB_dif) < 0.015

    """Test for time_frequency.Spectrogram()"""
    def _test_mono_valid():
        """Tests for
            - mono input
            - valid padding
            - shapes of output channel, n_freq, n_frame
            - save and load a model with it

        """
        n_ch = 1
        n_dft, len_hop, nsp_src = 512, 256, 8000
        src = np.random.uniform(-1., 1., nsp_src)

        model = tensorflow.keras.models.Sequential()
        model.add(Spectrogram(n_dft=n_dft, n_hop=len_hop, padding='valid',
                              power_spectrogram=1.0, return_decibel_spectrogram=False,
                              image_data_format='default',
                              input_shape=(n_ch, nsp_src)))
        batch_stft_kapre = model.predict(src[np.newaxis, np.newaxis, :])

        # check num_channel
        if image_data_format() == 'channels_last':
            assert batch_stft_kapre.shape[3] == n_ch
            assert batch_stft_kapre.shape[1] == n_dft // 2 + 1
            assert batch_stft_kapre.shape[2] == _num_frame_valid(nsp_src, n_dft, len_hop)
        else:
            assert batch_stft_kapre.shape[1] == n_ch
            assert batch_stft_kapre.shape[2] == n_dft // 2 + 1
            assert batch_stft_kapre.shape[3] == _num_frame_valid(nsp_src, n_dft, len_hop)

        # TODO: save the model

        # Now compare the result!
        # TODO. actually, later.
        # if image_data_format() == 'channels_last':
        #     S_kapre = batch_stft_kapre[0, :, :, 0]
        # else:
        #     S_kapre = batch_stft_kapre[0, 0, :, :]
        #
        # S_ref = librosa.stft(src, n_fft=n_dft, hop_length=len_hop)
        # S_ref = S_ref[:, 1:-1]
        # S_kapre = S_kapre[:, 1:-1]
        # assert np.allclose(S_kapre, np.abs(S_ref), atol=1e-5)

        # test power_spectrogram
        # test decibel
        # test if the kernel becomes trainable

    def _test_stereo_same():
        """Tests for
            - stereo input
            - same padding
            - shapes of output channel, n_freq, n_frame
            - save and load a model with it

        """
        n_ch = 2
        n_dft, len_hop, nsp_src = 512, 256, 8000
        src = np.random.uniform(-1., 1., (n_ch, nsp_src))

        model = tensorflow.keras.models.Sequential()
        model.add(Spectrogram(n_dft=n_dft, n_hop=len_hop, padding='same',
                              power_spectrogram=1.0, return_decibel_spectrogram=False,
                              image_data_format='default',
                              input_shape=(n_ch, nsp_src)))
        batch_stft_kapre = model.predict(src[np.newaxis, :])

        # check num_channel
        if image_data_format() == 'channels_last':
            assert batch_stft_kapre.shape[3] == n_ch
            assert batch_stft_kapre.shape[1] == n_dft // 2 + 1
            assert batch_stft_kapre.shape[2] == _num_frame_same(nsp_src, len_hop)
        else:
            assert batch_stft_kapre.shape[1] == n_ch
            assert batch_stft_kapre.shape[2] == n_dft // 2 + 1
            assert batch_stft_kapre.shape[3] == _num_frame_same(nsp_src, len_hop)

    K.set_image_data_format("channels_first")
    _test_mono_valid()
    _test_stereo_same()
    _test_correctness()
    K.set_image_data_format("channels_last")
    _test_mono_valid()
    _test_stereo_same()
    _test_correctness()


def test_melspectrogram():
    def _test_correctness():
        """ Tests correctness
        """
        audio_data, sr = librosa.load("speech_test_file.wav", sr=44100, mono=True)

        hop_length = 128
        n_fft = 1024
        n_mels = 80

        # compute with librosa
        S = librosa.feature.melspectrogram(audio_data, sr=sr, n_fft=n_fft, 
                                           hop_length=hop_length, 
                                           n_mels=n_mels)

        S_DB_librosa = librosa.power_to_db(S, ref=np.max)

        # compute with kapre
        mels_model = tensorflow.keras.models.Sequential()
        mels_model.add(Melspectrogram(sr=sr, n_mels=n_mels,
                                      n_dft=n_fft, n_hop=hop_length,
                                      input_shape=(1, len(audio_data)),
                                      power_melgram=2,
                                      return_decibel_melgram=False,
                                      trainable_kernel=False, name='melgram'))

        S = mels_model.predict(audio_data.reshape(1, 1, -1))
        if image_data_format() == 'channels_last':
            S = S[0, :, :, 0]
        else:
            S = S[0, 0]
        S_DB_kapre = librosa.power_to_db(S, ref=np.max)

        DB_scale = (np.max(S_DB_librosa) - np.min(S_DB_librosa))
        S_DB_dif = np.abs(S_DB_kapre - S_DB_librosa) / DB_scale
        assert np.mean(S_DB_dif) < 0.01

    """Test for time_frequency.Melspectrogram()"""
    def _test_mono_valid():
        """Tests for
            - mono input
            - valid padding
            - shapes of output channel, n_freq, n_frame
            - save and load a model with it

        """
        n_ch = 1
        sr = 12000
        n_mels = 96
        fmin, fmax = 0.0, sr // 2
        n_dft, len_hop, nsp_src = 512, 256, 12000
        src = np.random.uniform(-1., 1., nsp_src)

        model = tensorflow.keras.models.Sequential()
        model.add(Melspectrogram(sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax,
                                 n_dft=n_dft, n_hop=len_hop, padding='valid',
                                 power_melgram=1.0, return_decibel_melgram=False,
                                 image_data_format='default',
                                 input_shape=(n_ch, nsp_src)))
        batch_melgram_kapre = model.predict(src[np.newaxis, np.newaxis, :])
        if image_data_format() == 'channels_last':
            assert batch_melgram_kapre.shape[3] == n_ch
            assert batch_melgram_kapre.shape[1] == n_mels
            assert batch_melgram_kapre.shape[2] == _num_frame_valid(nsp_src, n_dft, len_hop)
        else:
            assert batch_melgram_kapre.shape[1] == n_ch
            assert batch_melgram_kapre.shape[2] == n_mels
            assert batch_melgram_kapre.shape[3] == _num_frame_valid(nsp_src, n_dft, len_hop)

    def _test_stereo_same():
        """Tests for
            - stereo input
            - same padding
            - shapes of output channel, n_freq, n_frame
            - save and load a model with it

        """
        n_ch = 2
        sr = 8000
        n_mels = 64
        fmin, fmax = 200, sr // 2
        n_dft, len_hop, nsp_src = 512, 256, 8000
        src = np.random.uniform(-1., 1., (n_ch, nsp_src))

        model = tensorflow.keras.models.Sequential()
        model.add(Melspectrogram(sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax,
                                 n_dft=n_dft, n_hop=len_hop, padding='same',
                                 power_melgram=1.0, return_decibel_melgram=False,
                                 image_data_format='default',
                                 input_shape=(n_ch, nsp_src)))
        batch_melgram_kapre = model.predict(src[np.newaxis, :])

        if image_data_format() == 'channels_last':
            assert batch_melgram_kapre.shape[3] == n_ch
            assert batch_melgram_kapre.shape[1] == n_mels
            assert batch_melgram_kapre.shape[2] == _num_frame_same(nsp_src, len_hop)
        else:
            assert batch_melgram_kapre.shape[1] == n_ch
            assert batch_melgram_kapre.shape[2] == n_mels
            assert batch_melgram_kapre.shape[3] == _num_frame_same(nsp_src, len_hop)

    K.set_image_data_format("channels_first")
    _test_mono_valid()
    _test_stereo_same()
    _test_correctness()

    K.set_image_data_format("channels_last")
    _test_mono_valid()
    _test_stereo_same()
    _test_correctness()


if __name__ == '__main__':
    pytest.main([__file__])
