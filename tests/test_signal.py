import pytest
import numpy as np
import tensorflow as tf
import librosa
import kapre
from kapre.backend import _CH_FIRST_STR, _CH_LAST_STR, _CH_DEFAULT_STR

from utils import get_audio


@pytest.mark.parametrize('data_format', ['default', 'channels_first', 'channels_last'])
@pytest.mark.parametrize('frame_length', [50, 32])
def test_frame_correctness(frame_length, data_format):
    hop_length = frame_length // 2
    n_ch = 1
    src_mono, batch_src, input_shape = get_audio(data_format=data_format, n_ch=n_ch, length=1000)

    model = tf.keras.Sequential()
    model.add(
        kapre.signal.Frame(
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


@pytest.mark.xfail()
def test_wrong_data_format():
    kapre.signal.Frame(32, 16, data_format='wrong_string')
