import os
import numpy as np
import tensorflow as tf
import tempfile
from tensorflow.keras import backend as K

SRC = np.load('tests/speech_test_file.npz')['audio_data'].astype(np.float32)


def get_audio(data_format, n_ch, length=8000):
    src = SRC
    src = src[:length]
    src_mono = src.copy()
    len_src = len(src)

    src = np.expand_dims(src, axis=1)  # (time, 1)
    if n_ch != 1:
        src = np.tile(src, [1, n_ch])  # (time, ch))

    if data_format == 'default':
        data_format = K.image_data_format()

    if data_format == 'channels_last':
        input_shape = (len_src, n_ch)
    else:
        src = np.transpose(src)  # (ch, time)
        input_shape = (n_ch, len_src)

    batch_src = np.expand_dims(src, axis=0)  # 3d batch input

    return src_mono, batch_src, input_shape


def save_load_compare(layer, input_batch, allclose_func, save_format, layer_class=None, training=None, atol=1e-4):
    """test a model with `layer` with the given `input_batch`.
    The model prediction result is compared using `allclose_func` which may depend on the
    data type of the model output (e.g., float or complex).
    """
    model = tf.keras.models.Sequential()
    model.add(layer)

    result_ref = model(input_batch, training=training)

    os_temp_dir = tempfile.gettempdir()
    model_temp_dir = tempfile.TemporaryDirectory(dir=os_temp_dir)

    if save_format == 'tf':
        model_path = model_temp_dir.name
    elif save_format == 'h5':
        model_path = os.path.join(model_temp_dir.name, 'model.h5')
    else:
        raise ValueError
    model.save(filepath=model_path, save_format=save_format)
    # if save_format == 'h5':
    #     import ipdb; ipdb.set_trace()

    if save_format == 'h5':
        new_model = tf.keras.models.load_model(
            model_path, custom_objects={layer.__class__.__name__: layer_class}
        )
    else:
        new_model = tf.keras.models.load_model(model_path)

    result_new = new_model(input_batch)
    allclose_func(result_ref, result_new, atol)

    model_temp_dir.cleanup()

    return model
