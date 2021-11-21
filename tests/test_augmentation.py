import pytest
import numpy as np
import tensorflow as tf
from kapre.augmentation import ChannelSwap, SpecAugment
from kapre.backend import _CH_FIRST_STR, _CH_LAST_STR, _CH_DEFAULT_STR

from utils import get_audio, get_spectrogram, save_load_compare


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
def test_spec_augment_shape_correctness(data_format):
    """
    Checks that shapes are correct depending on each data_format
    """

    batch_src, input_shape = get_spectrogram(data_format)

    model = tf.keras.Sequential()
    spec_augment = SpecAugment(
                               input_shape=input_shape,
                               freq_mask_param=5,
                               time_mask_param=10,
                               n_freq_masks=4,
                               n_time_masks=3,
                               mask_value=0.,
                               data_format=data_format)

    model.add(spec_augment)

    # We must force training to True to test properly if SpecAugment works as expected
    spec_augmented = model(batch_src, training=True)[0]

    np.testing.assert_equal(model.layers[0].output_shape[1:], spec_augmented.shape)


def test_spec_augment_exception():
    """
    Checks that SpecAugments fails if Spectrogram has depth greater than 1.
    """

    data_format = "default"
    with pytest.raises(RuntimeError):

        batch_src, input_shape = get_spectrogram(data_format=data_format, n_ch=4)

        model = tf.keras.Sequential()
        spec_augment = SpecAugment(
            input_shape=input_shape,
            freq_mask_param=5,
            time_mask_param=10,
            data_format=data_format)
        model.add(spec_augment)
        _ = model(batch_src, training=True)[0]


@pytest.mark.parametrize('data_format', ['default', 'channels_first', 'channels_last'])
@pytest.mark.parametrize('save_format', ['tf', 'h5'])
def test_save_load_channel_swap(data_format, save_format):
    src_mono, batch_src, input_shape = get_audio(data_format='channels_last', n_ch=1)

    save_load_compare(
        ChannelSwap(input_shape=input_shape),
        batch_src,
        np.testing.assert_allclose,
        save_format=save_format,
        layer_class=ChannelSwap,
        training=None,
    )


@pytest.mark.parametrize('data_format', ['default', 'channels_first', 'channels_last'])
@pytest.mark.parametrize('save_format', ['tf', 'h5'])
def test_save_load_spec_augment(data_format, save_format):
    batch_src, input_shape = get_spectrogram(data_format='channels_last', n_ch=1)

    save_load_compare(
        SpecAugment(input_shape=input_shape, freq_mask_param=3, time_mask_param=5),
        batch_src,
        np.testing.assert_allclose,
        save_format=save_format,
        layer_class=SpecAugment,
        training=None,
    )
