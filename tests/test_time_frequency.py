import pytest
import numpy as np
import keras
import keras.backend as K
from keras.backend import image_data_format
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

        model = keras.models.Sequential()
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

        model = keras.models.Sequential()
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
    K.set_image_data_format("channels_last")
    _test_mono_valid()
    _test_stereo_same()


def test_melspectrogram():
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

        model = keras.models.Sequential()
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

        model = keras.models.Sequential()
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

    K.set_image_data_format("channels_last")
    _test_mono_valid()
    _test_stereo_same()


if __name__ == '__main__':
    pytest.main([__file__])
