import pytest
import numpy as np
import tensorflow as tf
import librosa
from kapre.signal import Frame, Energy, MuLawEncoding, MuLawDecoding
from kapre.backend import _CH_FIRST_STR, _CH_LAST_STR, _CH_DEFAULT_STR

from utils import get_audio, save_load_compare


@pytest.mark.parametrize('data_format', ['default', 'channels_first', 'channels_last'])
@pytest.mark.parametrize('frame_length', [50, 32])
def test_frame_correctness(frame_length, data_format):
    hop_length = frame_length // 2
    n_ch = 1
    src_mono, batch_src, input_shape = get_audio(data_format=data_format, n_ch=n_ch, length=1000)

    model = tf.keras.Sequential()
    model.add(
        Frame(
            frame_length=frame_length,
            hop_length=hop_length,
            pad_end=False,
            data_format=data_format,
            input_shape=input_shape,
        )
    )

    frames_ref = librosa.util.frame(src_mono, frame_length, hop_length).T  # (time, frame_length)

    if data_format in (_CH_DEFAULT_STR, _CH_LAST_STR):
        frames_ref = np.expand_dims(frames_ref, axis=2)
    else:
        frames_ref = np.expand_dims(frames_ref, axis=0)

    frames_kapre = model.predict(batch_src)[0]

    np.testing.assert_equal(frames_kapre, frames_ref)


@pytest.mark.parametrize('data_format', ['default', 'channels_first', 'channels_last'])
def test_energy_correctness(data_format):
    frame_length = 4
    hop_length = frame_length // 2
    n_ch = 1
    src_mono, batch_src, input_shape = get_audio(
        data_format=data_format, n_ch=n_ch, length=frame_length * 2
    )

    sr = 22050
    ref_duration = 0.1
    model = tf.keras.Sequential()
    model.add(
        Energy(
            sample_rate=sr,
            ref_duration=ref_duration,
            frame_length=frame_length,
            hop_length=hop_length,
            pad_end=False,
            data_format=data_format,
            input_shape=input_shape,
        )
    )

    energies_kapre = model.predict(batch_src)[0]

    frames_ref = librosa.util.frame(src_mono, frame_length, hop_length).T  # (time, frame_length)
    nor_coeff = ref_duration / (frame_length / sr)
    energies_ref = nor_coeff * np.sum(frames_ref ** 2, axis=1)  # (time, )

    if data_format in (_CH_DEFAULT_STR, _CH_LAST_STR):
        energies_ref = np.expand_dims(energies_ref, axis=1)
    else:
        energies_ref = np.expand_dims(energies_ref, axis=0)

    np.testing.assert_allclose(energies_kapre, energies_ref, atol=1e-5)


def test_save_load():
    src_mono, batch_src, input_shape = get_audio(data_format='channels_last', n_ch=1)
    # test Frame save/load
    save_load_compare(
        Frame(frame_length=128, hop_length=64, input_shape=input_shape),
        batch_src,
        np.testing.assert_allclose,
    )
    # test Energy save/load
    save_load_compare(
        Energy(frame_length=128, hop_length=64, input_shape=input_shape),
        batch_src,
        np.testing.assert_allclose,
    )
    # test mu law layers
    save_load_compare(
        MuLawEncoding(quantization_channels=256), batch_src, np.testing.assert_allclose,
    )
    save_load_compare(
        MuLawDecoding(quantization_channels=256),
        np.arange(0, 256, 1).reshape((1, 256, 1)),
        np.testing.assert_allclose,
    )


@pytest.mark.xfail()
def test_wrong_data_format():
    Frame(32, 16, data_format='wrong_string')
