import numpy as np
from tensorflow.keras import backend as K

SRC = np.load('tests/speech_test_file.npz')['audio_data']


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
