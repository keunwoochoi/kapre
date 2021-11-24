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


@pytest.mark.parametrize('inputs', [("default", 0, 5), ("default", 3, 5), ("default", 0, 500),
                                    ("channels_last", 1, 5), ("channels_last", 3, 5), ("channels_last", 1, 500),
                                    ("channels_first", 2, 5), ("channels_first", 3, 5), ("channels_first", 2, 500)])
def test_spec_augment_generate_axis_mask(inputs):
    """
    Tests the static method _generate_axis_mask to see if shape is kept and
    exceptions are caught
    """

    data_format, axis, mask_param = inputs
    batch_src, input_shape = get_spectrogram(data_format)

    spec_augment = SpecAugment(
        input_shape=input_shape,
        freq_mask_param=5,
        time_mask_param=10,
        n_freq_masks=4,
        n_time_masks=3,
        mask_value=0.,
        data_format=data_format)

    # We force axis that will trigger NotImplementedError
    if axis not in [0, 1, 2]:
        # Check axis error
        with pytest.raises(NotImplementedError):
            # We use batch_src instead of batch_src[0] to simulate a 4D spectrogram
            inputs = (batch_src, axis, mask_param)
            spec_augment._generate_axis_mask(inputs)

    # We force mask_params that will trigger the ValueError. If it is not triggered, then
    # inputs are ok, so we must only test if the shapes are kept during transformation
    elif mask_param != 5:
        # Check mask_param error
        with pytest.raises(ValueError):
            inputs = (batch_src[0], axis, mask_param)
            spec_augment._generate_axis_mask(inputs)
    else:
        # Check that transformation keeps shape
        inputs = (batch_src[0], axis, mask_param)
        mask = spec_augment._generate_axis_mask(inputs)
        np.testing.assert_equal(mask.shape[axis], input_shape[axis])


@pytest.mark.parametrize('data_format_and_axis', [('default', 0), ('channels_last', 1), ('channels_first', 2)])
@pytest.mark.parametrize('mask_param', [10, 12, 20])
@pytest.mark.parametrize('n_masks', [1, 5, 10])
def test_spec_augment_apply_masks_to_axis(data_format_and_axis, mask_param, n_masks):
    """
    Tests method _apply_masks_to_axis, which applied a number of masks defined by
    parameter n_masks to the axis provided
    """

    data_format, axis = data_format_and_axis
    batch_src, input_shape = get_spectrogram(data_format)

    spec_augment = SpecAugment(
        input_shape=input_shape,
        freq_mask_param=5,
        time_mask_param=10,
        n_freq_masks=4,
        n_time_masks=3,
        mask_value=0.,
        data_format=data_format)

    spec_masked = spec_augment._apply_masks_to_axis(batch_src[0], axis, mask_param, n_masks)
    np.testing.assert_equal(spec_masked.shape, input_shape)


@pytest.mark.parametrize('data_format', ['default', 'channels_first', 'channels_last'])
def test_spec_augment_layer(data_format):
    """
    Tests the complete layer, checking if the parameter `training` has the expected behaviour.
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

    # Fist, enforce training to True and check the shapes
    spec_augmented = model(batch_src, training=True)
    np.testing.assert_equal(model.layers[0].output_shape[1:], spec_augmented[0].shape)

    # Second, check that it doesn't change anything in default
    spec_augmented = model(batch_src)
    np.testing.assert_allclose(spec_augmented, batch_src)

# def test_spec_augment_depth_exception():
#     """
#     Checks that SpecAugments fails if Spectrogram has depth greater than 1.
#     """
#
#     data_format = "default"
#     with pytest.raises(RuntimeError):
#
#         batch_src, input_shape = get_spectrogram(data_format=data_format, n_ch=4)
#
#         model = tf.keras.Sequential()
#         spec_augment = SpecAugment(
#             input_shape=input_shape,
#             freq_mask_param=5,
#             time_mask_param=10,
#             data_format=data_format)
#         model.add(spec_augment)
#         _ = model(batch_src, training=True)[0]
#
#
# @pytest.mark.parametrize('data_format', ['default', 'channels_first', 'channels_last'])
# @pytest.mark.parametrize('save_format', ['tf', 'h5'])
# def test_save_load_channel_swap(data_format, save_format):
#     src_mono, batch_src, input_shape = get_audio(data_format='channels_last', n_ch=1)
#
#     save_load_compare(
#         ChannelSwap(input_shape=input_shape),
#         batch_src,
#         np.testing.assert_allclose,
#         save_format=save_format,
#         layer_class=ChannelSwap,
#         training=None,
#     )
#
#
# @pytest.mark.parametrize('data_format', ['default', 'channels_first', 'channels_last'])
# @pytest.mark.parametrize('save_format', ['tf', 'h5'])
# def test_save_load_spec_augment(data_format, save_format):
#     batch_src, input_shape = get_spectrogram(data_format='channels_last', n_ch=1)
#
#     save_load_compare(
#         SpecAugment(input_shape=input_shape, freq_mask_param=3, time_mask_param=5),
#         batch_src,
#         np.testing.assert_allclose,
#         save_format=save_format,
#         layer_class=SpecAugment,
#         training=None,
#     )
