import pytest
import numpy as np
import tensorflow as tf
from kapre.augmentation import ChannelSwap
from kapre.backend import _CH_FIRST_STR, _CH_LAST_STR, _CH_DEFAULT_STR

from utils import get_audio, save_load_compare


@pytest.mark.parametrize('data_format', ['default', 'channels_first', 'channels_last'])
@pytest.mark.parametrize('n_ch', [1, 4])
@pytest.mark.parametrize('data_type', ['1d'])  # wip
def test_channel_swap_correctness(n_ch, data_format, data_type):
    len_src = 256
    src_mono, batch_src, input_shape = get_audio(data_format=data_format, n_ch=n_ch, length=len_src)

    model = tf.keras.Sequential()
    model.add(
        ChannelSwap(
            input_shape=input_shape,
        )
    )
    # consistent during inference
    kapre_ref = model.predict(batch_src)
    for _ in range(100):
        kapre_again = model.predict(batch_src)
        np.testing.assert_equal(kapre_ref, kapre_again)
    ch_axis = 1 if data_format == _CH_FIRST_STR else 2  # to be changed for 2d data type

    # during training.. --> todo
    # if n_ch == 1:
    #     return
    #
    # kapre_augs = []
    # for _ in range(n_ch ** 2):
    #     kapre_augs.append(model(batch_src, training=True))


@pytest.mark.parametrize('data_format', ['default', 'channels_first', 'channels_last'])
@pytest.mark.parametrize('save_format', ['tf', 'h5'])
def test_save_load(data_format, save_format):
    src_mono, batch_src, input_shape = get_audio(data_format='channels_last', n_ch=1)

    save_load_compare(
        ChannelSwap(input_shape=input_shape),
        batch_src,
        np.testing.assert_allclose,
        save_format=save_format,
        layer_class=ChannelSwap,
        training=None,
    )
