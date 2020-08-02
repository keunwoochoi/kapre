import pytest
import numpy as np
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.backend import image_data_format
import librosa
from kapre.time_frequency import STFT, Melspectrogram


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
    if image_data_format() == 'channels_last':
        S = S[0, :, :, 0]
    else:
        S = S[0, 0]

    S_DB_librosa = librosa.power_to_db(S_librosa, ref=np.max)

    # magnitudes_kapre = librosa.magphase(S, power=1)[0]
    # S_DB_kapre = librosa.power_to_db(magnitudes_kapre, ref=np.max)
    #
    # DB_scale = np.max(S_DB_librosa) - np.min(S_DB_librosa)
    # S_DB_dif = np.abs(S_DB_kapre - S_DB_librosa) / DB_scale
    #
    # assert np.allclose(magnitudes_expected, magnitudes_kapre, rtol=1e-2, atol=1e-8)
    # assert np.mean(S_DB_dif) < 0.015


# def test_melspectrogram():
#     def _test_correctness():
#         """ Tests correctness
#         """
#         audio_data = np.load('tests/speech_test_file.npz')['audio_data']
#         sr = 44100
#
#         hop_length = 128
#         n_fft = 1024
#         n_mels = 80
#
#         # compute with librosa
#         S = librosa.feature.melspectrogram(
#             audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
#         )
#
#         S_DB_librosa = librosa.power_to_db(S, ref=np.max)
#
#         # load precomputed
#         S_expected = np.load('tests/test_audio_mel_g0.npy')
#
#         # compute with kapre
#         mels_model = tensorflow.keras.models.Sequential()
#         mels_model.add(
#             Melspectrogram(
#                 sr=sr,
#                 n_mels=n_mels,
#                 n_dft=n_fft,
#                 n_hop=hop_length,
#                 input_shape=(1, len(audio_data)),
#                 power_melgram=2,
#                 return_decibel_melgram=False,
#                 trainable_kernel=False,
#                 name='melgram',
#             )
#         )
#
#         S = mels_model.predict(audio_data.reshape(1, 1, -1))
#         if image_data_format() == 'channels_last':
#             S = S[0, :, :, 0]
#         else:
#             S = S[0, 0]
#         S_DB_kapre = librosa.power_to_db(S, ref=np.max)
#
#         DB_scale = np.max(S_DB_librosa) - np.min(S_DB_librosa)
#         S_DB_dif = np.abs(S_DB_kapre - S_DB_librosa) / DB_scale
#
#         # compare expected float32 values with computed ones
#         assert np.allclose(S_expected, S, rtol=1e-2, atol=1e-8)
#         assert np.mean(S_DB_dif) < 0.01
#
#     """Test for time_frequency.Melspectrogram()"""
#
#     def _test_mono_valid():
#         """Tests for
#             - mono input
#             - valid padding
#             - shapes of output channel, n_freq, n_frame
#             - save and load a model with it
#
#         """
#         n_ch = 1
#         sr = 12000
#         n_mels = 96
#         fmin, fmax = 0.0, sr // 2
#         n_dft, len_hop, nsp_src = 512, 256, 12000
#         src = np.random.uniform(-1.0, 1.0, nsp_src)
#
#         model = tensorflow.keras.models.Sequential()
#         model.add(
#             Melspectrogram(
#                 sr=sr,
#                 n_mels=n_mels,
#                 fmin=fmin,
#                 fmax=fmax,
#                 n_dft=n_dft,
#                 n_hop=len_hop,
#                 padding='valid',
#                 power_melgram=1.0,
#                 return_decibel_melgram=False,
#                 image_data_format='default',
#                 input_shape=(n_ch, nsp_src),
#             )
#         )
#         batch_melgram_kapre = model.predict(src[np.newaxis, np.newaxis, :])
#         if image_data_format() == 'channels_last':
#             assert batch_melgram_kapre.shape[3] == n_ch
#             assert batch_melgram_kapre.shape[1] == n_mels
#             assert batch_melgram_kapre.shape[2] == _num_frame_valid(nsp_src, n_dft, len_hop)
#         else:
#             assert batch_melgram_kapre.shape[1] == n_ch
#             assert batch_melgram_kapre.shape[2] == n_mels
#             assert batch_melgram_kapre.shape[3] == _num_frame_valid(nsp_src, n_dft, len_hop)
#
#     def _test_stereo_same():
#         """Tests for
#             - stereo input
#             - same padding
#             - shapes of output channel, n_freq, n_frame
#             - save and load a model with it
#
#         """
#         n_ch = 2
#         sr = 8000
#         n_mels = 64
#         fmin, fmax = 200, sr // 2
#         n_dft, len_hop, nsp_src = 512, 256, 8000
#         src = np.random.uniform(-1.0, 1.0, (n_ch, nsp_src))
#
#         model = tensorflow.keras.models.Sequential()
#         model.add(
#             Melspectrogram(
#                 sr=sr,
#                 n_mels=n_mels,
#                 fmin=fmin,
#                 fmax=fmax,
#                 n_dft=n_dft,
#                 n_hop=len_hop,
#                 padding='same',
#                 power_melgram=1.0,
#                 return_decibel_melgram=False,
#                 image_data_format='default',
#                 input_shape=(n_ch, nsp_src),
#             )
#         )
#         batch_melgram_kapre = model.predict(src[np.newaxis, :])
#
#         if image_data_format() == 'channels_last':
#             assert batch_melgram_kapre.shape[3] == n_ch
#             assert batch_melgram_kapre.shape[1] == n_mels
#             assert batch_melgram_kapre.shape[2] == _num_frame_same(nsp_src, len_hop)
#         else:
#             assert batch_melgram_kapre.shape[1] == n_ch
#             assert batch_melgram_kapre.shape[2] == n_mels
#             assert batch_melgram_kapre.shape[3] == _num_frame_same(nsp_src, len_hop)
#
#     K.set_image_data_format("channels_first")
#     _test_mono_valid()
#     _test_stereo_same()
#     _test_correctness()
#
#     K.set_image_data_format("channels_last")
#     _test_mono_valid()
#     _test_stereo_same()
#     _test_correctness()


if __name__ == '__main__':
    pytest.main([__file__])
