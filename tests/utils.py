import os
from pathlib import Path
import shutil
import numpy as np
import tensorflow as tf
import tempfile
from tensorflow.keras import backend as K
from kapre import backend

SRC = np.load('tests/speech_test_file.npz')['audio_data'].astype(np.float32)


def get_audio(data_format, n_ch, length=8000, batch_size=1):
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

    # batch_src = np.expand_dims(src, axis=0)  # 3d batch input
    batch_src = np.repeat([src], batch_size, axis=0)

    return src_mono, batch_src, input_shape


def get_spectrogram(data_format, n_ch=1, time_dimension=256, freq_dimension=128, batch_size=1):
    src = np.random.uniform(low=-10, high=10, size=(time_dimension, freq_dimension))

    src = np.expand_dims(src, axis=2)  # (time, freq, 1)
    if n_ch != 1:
        src = np.tile(src, [1, n_ch])  # (time, freq, ch))

    if data_format == 'default':
        data_format = K.image_data_format()

    if data_format == 'channels_last':
        input_shape = (time_dimension, freq_dimension, n_ch)
    else:
        src = np.transpose(src, (2, 0, 1))  # (ch, time, freq)
        input_shape = (n_ch, time_dimension, freq_dimension)

    batch_src = np.repeat([src], batch_size, axis=0)

    return batch_src, input_shape


def save_load_compare(
    layer,
    input_batch,
    assertion_callback,
    save_format='tf',
    layer_class=None,
    training=None,
    input_shape=None,
):
    """
    Save a model with `layer` and load it back, and compare the outputs.
    The model prediction result is compared using `allclose_func` which may depend on the
    data type of the model output (e.g., float or complex).
    """
    if not isinstance(layer, tf.keras.Model):
        model = tf.keras.Sequential()
        if input_shape is not None:
            model.add(tf.keras.Input(shape=input_shape))
        model.add(layer)
    else:
        model = layer

    if training is None:
        result_original = model.predict(input_batch)
    else:
        result_original = model(input_batch, training=training)

    os_temp_dir = tempfile.gettempdir()
    model_temp_dir = tempfile.TemporaryDirectory(dir=os_temp_dir)

    if save_format == 'tf':
        model_path = os.path.join(model_temp_dir.name, 'model.keras')
    elif save_format == 'h5':
        model_path = os.path.join(model_temp_dir.name, 'model.h5')
    else:
        raise ValueError
    model.save(model_path)
    # if save_format == 'h5':
    #     import ipdb; ipdb.set_trace()

    if save_format == 'h5':
        new_model = tf.keras.models.load_model(
            model_path, custom_objects={layer.__class__.__name__: layer_class}
        )
    else:
        new_model = tf.keras.models.load_model(model_path)

    if training is None:
        result_new = new_model.predict(input_batch)
    else:
        result_new = new_model(input_batch, training=training)

    assertion_callback(result_original, result_new)

    model_temp_dir.cleanup()

    return model


def predict_using_tflite(model, batch_src):
    """Convert a keras model to tflite and infer on batch_src

    Attempts to convert a keras model to a tflite model, load the tflite model,
    then infer on the data in batch_src
    Args:
        model (keras model)
        batch_src (numpy array) - audio to test model
    Returns:
        pred_tflite (numpy array) - array of predictions.
    """
    ############################################################################
    # TF lite conversion
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.SELECT_TF_OPS,
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    tflite_model = converter.convert()
    model_name = 'test_tflite'
    path = Path("/tmp/tflite_tests/")
    # make a temporary location
    if path.exists():
        shutil.rmtree(path)
    os.makedirs(path)
    tflite_file = path / Path(model_name + ".tflite")
    open(tflite_file.as_posix(), "wb").write(tflite_model)

    ############################################################################
    # Make sure we can load and infer on the TFLITE model
    interpreter = tf.lite.Interpreter(tflite_file.as_posix())
    # infer on each input separately and collect the predictions
    pred_tflite = []

    for x in batch_src:

        # set batch size for tflite
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # apply input tensors, expand first dimension to create batch dimension
        interpreter.set_tensor(input_details[0]["index"], np.expand_dims(x, 0))
        # infer
        interpreter.invoke()
        tflite_results = interpreter.get_tensor(output_details[0]["index"])

        pred_tflite.append(tflite_results)

    return np.concatenate(pred_tflite, axis=0)
