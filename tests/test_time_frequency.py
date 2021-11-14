import pytest
import numpy as np
import tensorflow as tf
import tensorflow.keras
import librosa
from kapre import (
    STFT,
    Magnitude,
    Phase,
    Delta,
    InverseSTFT,
    ApplyFilterbank,
    ConcatenateFrequencyMap,
    STFTTflite,
    MagnitudeTflite,
    PhaseTflite,
)
from kapre.composed import (
    get_melspectrogram_layer,
    get_log_frequency_spectrogram_layer,
    get_stft_mag_phase,
    get_perfectly_reconstructing_stft_istft,
    get_stft_magnitude_layer,
    get_frequency_aware_conv2d,
)

from utils import get_audio, save_load_compare, predict_using_tflite


def _num_frame_valid(nsp_src, nsp_win, len_hop):
    """Computes the number of frames with 'valid' setting"""
    return (nsp_src - (nsp_win - len_hop)) // len_hop


def _num_frame_same(nsp_src, len_hop):
    """Computes the number of frames with 'same' setting"""
    return int(np.ceil(float(nsp_src) / len_hop))


def allclose_phase(a, b, atol=1e-3):
    """Testing phase.
    Remember that a small error in complex value may lead to a large phase difference
    if the norm is very small.

    Therefore, it makes more sense to test it on the complex value itself rather than breaking it down to phase.

    """
    np.testing.assert_allclose(np.sin(a), np.sin(b), atol=atol)
    np.testing.assert_allclose(np.cos(a), np.cos(b), atol=atol)


def assert_approx_phase(a, b, atol=1e-2, acceptable_fail_ratio=0.01):
    """Testing approximate phase.
    Tflite phase is approximate, some values will allways have a large error
    So makes more sense to count the number that are within tolerance
    """
    count_failed = np.sum(np.abs(a - b) > atol)
    assert (
        count_failed / a.size < acceptable_fail_ratio
    ), "too many inaccuracte phase bins: {} bins out of {} incorrect".format(count_failed, a.size)


def allclose_complex_numbers(a, b, atol=1e-3):
    np.testing.assert_equal(np.shape(a), np.shape(b))
    np.testing.assert_allclose(np.abs(a), np.abs(b), rtol=1e-5, atol=atol)
    np.testing.assert_allclose(np.real(a), np.real(b), rtol=1e-5, atol=atol)
    np.testing.assert_allclose(np.imag(a), np.imag(b), rtol=1e-5, atol=atol)


@pytest.mark.parametrize('n_fft', [1000])
@pytest.mark.parametrize('hop_length', [None, 256])
@pytest.mark.parametrize('n_ch', [1, 2, 6])
@pytest.mark.parametrize('data_format', ['default', 'channels_first', 'channels_last'])
@pytest.mark.parametrize('batch_size', [1, 10])
def test_spectrogram_correctness(n_fft, hop_length, n_ch, data_format, batch_size):
    def _get_stft_model(following_layer=None):
        # compute with kapre
        stft_model = tensorflow.keras.models.Sequential()
        stft_model.add(
            STFT(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                window_name=None,
                pad_end=False,
                input_data_format=data_format,
                output_data_format=data_format,
                input_shape=input_shape,
                name='stft',
            )
        )
        if following_layer is not None:
            stft_model.add(following_layer)
        return stft_model

    src_mono, batch_src, input_shape = get_audio(
        data_format=data_format, n_ch=n_ch, batch_size=batch_size
    )
    win_length = n_fft  # test with x2
    # compute with librosa
    S_ref = librosa.core.stft(
        src_mono, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False
    ).T  # (time, freq)

    S_ref = np.expand_dims(S_ref, axis=2)  # time, freq, ch=1
    S_ref = np.tile(S_ref, [1, 1, n_ch])  # time, freq, ch=n_ch
    if data_format == 'channels_first':
        S_ref = np.transpose(S_ref, (2, 0, 1))  # ch, time, freq

    stft_model = _get_stft_model()

    S_complex = stft_model.predict(batch_src)[0]  # 3d representation
    allclose_complex_numbers(S_ref, S_complex)

    # test Magnitude()
    stft_mag_model = _get_stft_model(Magnitude())
    S = stft_mag_model.predict(batch_src)[0]  # 3d representation
    np.testing.assert_allclose(np.abs(S_ref), S, atol=2e-4)

    # # test Phase()
    stft_phase_model = _get_stft_model(Phase())
    S = stft_phase_model.predict(batch_src)[0]  # 3d representation
    allclose_phase(np.angle(S_complex), S)


@pytest.mark.parametrize('data_format', ['channels_first', 'channels_last'])
@pytest.mark.parametrize('window_name', [None, 'hann_window', 'hamming_window'])
def test_spectrogram_correctness_more(data_format, window_name):
    def _get_stft_model(following_layer=None):
        # compute with kapre
        stft_model = tensorflow.keras.models.Sequential()
        stft_model.add(
            STFT(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                window_name=window_name,
                pad_end=False,
                input_data_format=data_format,
                output_data_format=data_format,
                input_shape=input_shape,
                name='stft',
            )
        )
        if following_layer is not None:
            stft_model.add(following_layer)
        return stft_model

    n_fft = 512
    hop_length = 256
    n_ch = 2

    src_mono, batch_src, input_shape = get_audio(data_format=data_format, n_ch=n_ch)
    win_length = n_fft  # test with x2
    # compute with librosa
    S_ref = librosa.core.stft(
        src_mono,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=False,
        window=window_name.replace('_window', '') if window_name else 'hann',
    ).T  # (time, freq)

    S_ref = np.expand_dims(S_ref, axis=2)  # time, freq, ch=1
    S_ref = np.tile(S_ref, [1, 1, n_ch])  # time, freq, ch=n_ch
    if data_format == 'channels_first':
        S_ref = np.transpose(S_ref, (2, 0, 1))  # ch, time, freq

    stft_model = _get_stft_model()

    S_complex = stft_model.predict(batch_src)[0]  # 3d representation
    allclose_complex_numbers(S_ref, S_complex)

    # test Magnitude()
    stft_mag_model = _get_stft_model(Magnitude())
    S = stft_mag_model.predict(batch_src)[0]  # 3d representation
    np.testing.assert_allclose(np.abs(S_ref), S, atol=2e-4)

    # # test Phase()
    stft_phase_model = _get_stft_model(Phase())
    S = stft_phase_model.predict(batch_src)[0]  # 3d representation
    allclose_phase(np.angle(S_complex), S)


@pytest.mark.parametrize('n_fft', [512])
@pytest.mark.parametrize('sr', [22050])
@pytest.mark.parametrize('hop_length', [None, 256])
@pytest.mark.parametrize('n_ch', [2])
@pytest.mark.parametrize('data_format', ['default', 'channels_first', 'channels_last'])
@pytest.mark.parametrize('amin', [1e-5, 1e-3])
@pytest.mark.parametrize('dynamic_range', [120.0, 80.0])
@pytest.mark.parametrize('n_mels', [40])
@pytest.mark.parametrize('mel_f_min', [0.0])
@pytest.mark.parametrize('mel_f_max', [8000])
def test_melspectrogram_correctness(
    n_fft, sr, hop_length, n_ch, data_format, amin, dynamic_range, n_mels, mel_f_min, mel_f_max
):
    """Test the correctness of melspectrogram.

    Note that mel filterbank is tested separated

    """

    def _get_melgram_model(return_decibel, amin, dynamic_range, input_shape=None):
        # compute with kapre
        melgram_model = get_melspectrogram_layer(
            n_fft=n_fft,
            sample_rate=sr,
            n_mels=n_mels,
            mel_f_min=mel_f_min,
            mel_f_max=mel_f_max,
            win_length=win_length,
            hop_length=hop_length,
            input_data_format=data_format,
            output_data_format=data_format,
            return_decibel=return_decibel,
            input_shape=input_shape,
            db_amin=amin,
            db_dynamic_range=dynamic_range,
        )
        return melgram_model

    src_mono, batch_src, input_shape = get_audio(data_format=data_format, n_ch=n_ch)

    win_length = n_fft  # test with x2
    # compute with librosa
    S_ref = librosa.feature.melspectrogram(
        src_mono,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        center=False,
        power=1.0,
        n_mels=n_mels,
        fmin=mel_f_min,
        fmax=mel_f_max,
    ).T

    S_ref = np.expand_dims(S_ref, axis=2)  # time, freq, ch=1
    S_ref = np.tile(S_ref, [1, 1, n_ch])  # time, freq, ch=n_ch

    if data_format == 'channels_first':
        S_ref = np.transpose(S_ref, (2, 0, 1))  # ch, time, freq

    # melgram
    melgram_model = _get_melgram_model(
        return_decibel=False, input_shape=input_shape, amin=None, dynamic_range=120.0
    )
    S = melgram_model.predict(batch_src)[0]  # 3d representation
    np.testing.assert_allclose(S_ref, S, atol=1e-4)

    # log melgram
    melgram_model = _get_melgram_model(
        return_decibel=True, input_shape=input_shape, amin=amin, dynamic_range=dynamic_range
    )
    S = melgram_model.predict(batch_src)[0]  # 3d representation
    S_ref_db = librosa.power_to_db(S_ref, ref=1.0, amin=amin, top_db=dynamic_range)

    np.testing.assert_allclose(
        S_ref_db, S, rtol=3e-3
    )  # decibel is evaluated with relative tolerance


@pytest.mark.parametrize('n_fft', [1000])
@pytest.mark.parametrize('hop_length', [None, 256])
@pytest.mark.parametrize('n_ch', [1, 2])
@pytest.mark.parametrize('data_format', ['default', 'channels_first', 'channels_last'])
@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('win_length', [1000, 512])
@pytest.mark.parametrize('pad_end', [False, True])
def test_spectrogram_tflite_correctness(
    n_fft, hop_length, n_ch, data_format, batch_size, win_length, pad_end
):
    def _get_stft_model(following_layer=None, tflite_compatible=False):
        # compute with kapre
        stft_model = tensorflow.keras.models.Sequential()
        if tflite_compatible:
            stft_model.add(
                STFTTflite(
                    n_fft=n_fft,
                    win_length=win_length,
                    hop_length=hop_length,
                    window_name=None,
                    pad_end=pad_end,
                    input_data_format=data_format,
                    output_data_format=data_format,
                    input_shape=input_shape,
                    name='stft',
                )
            )
        else:
            stft_model.add(
                STFT(
                    n_fft=n_fft,
                    win_length=win_length,
                    hop_length=hop_length,
                    window_name=None,
                    pad_end=pad_end,
                    input_data_format=data_format,
                    output_data_format=data_format,
                    input_shape=input_shape,
                    name='stft',
                )
            )
        if following_layer is not None:
            stft_model.add(following_layer)
        return stft_model

    src_mono, batch_src, input_shape = get_audio(
        data_format=data_format, n_ch=n_ch, batch_size=batch_size
    )
    # tflite requires a known batch size
    batch_size = batch_src.shape[0]

    stft_model_tflite = _get_stft_model(tflite_compatible=True)
    stft_model = _get_stft_model(tflite_compatible=False)

    # test STFT()
    S_complex_tflite = predict_using_tflite(stft_model_tflite, batch_src)  # predict using tflite
    # (batch, time, freq, chan, re/imag) - convert to complex number:
    S_complex_tflite = tf.complex(
        S_complex_tflite[..., 0], S_complex_tflite[..., 1]
    )  # (batch,time,freq,chan)
    S_complex = stft_model.predict(batch_src)  # predict using tf model
    allclose_complex_numbers(S_complex, S_complex_tflite)

    # test Magnitude()
    stft_mag_model_tflite = _get_stft_model(MagnitudeTflite(), tflite_compatible=True)
    stft_mag_model = _get_stft_model(Magnitude(), tflite_compatible=False)
    S_lite = predict_using_tflite(stft_mag_model_tflite, batch_src)  # predict using tflite
    S = stft_mag_model.predict(batch_src)  # predict using tf model
    np.testing.assert_allclose(S, S_lite, atol=1e-4)

    # # test approx Phase() same for tflite and non-tflite
    stft_approx_phase_model_lite = _get_stft_model(
        PhaseTflite(approx_atan_accuracy=500), tflite_compatible=True
    )
    stft_approx_phase_model = _get_stft_model(
        Phase(approx_atan_accuracy=500), tflite_compatible=False
    )
    S_approx_phase_lite = predict_using_tflite(
        stft_approx_phase_model_lite, batch_src
    )  # predict using tflite
    S_approx_phase = stft_approx_phase_model.predict(
        batch_src, batch_size=batch_size
    )  # predict using tf model
    assert_approx_phase(S_approx_phase_lite, S_approx_phase, atol=1e-2, acceptable_fail_ratio=0.01)

    # # test accuracy of approx Phase()
    stft_phase_model = _get_stft_model(Phase(), tflite_compatible=False)
    S_phase = stft_phase_model.predict(batch_src, batch_size=batch_size)  # predict using tf model
    assert_approx_phase(S_approx_phase_lite, S_phase, atol=1e-2, acceptable_fail_ratio=0.01)


@pytest.mark.parametrize('data_format', ['default', 'channels_first', 'channels_last'])
def test_log_spectrogram_runnable(data_format):
    """test if log spectrogram layer works well"""
    src_mono, batch_src, input_shape = get_audio(data_format=data_format, n_ch=1)
    _ = get_log_frequency_spectrogram_layer(input_shape, return_decibel=True)
    _ = get_log_frequency_spectrogram_layer(input_shape, return_decibel=False)


@pytest.mark.xfail
def test_log_spectrogram_fail():
    """test if log spectrogram layer works well"""
    src_mono, batch_src, input_shape = get_audio(data_format='channels_last', n_ch=1)
    _ = get_log_frequency_spectrogram_layer(input_shape, return_decibel=True, log_n_bins=200)


def test_delta():
    """test delta layer"""
    specgrams = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    specgrams = np.reshape(specgrams, (1, -1, 1, 1))  # (b, t, f, ch)
    delta_model = tensorflow.keras.models.Sequential()
    delta_model.add(Delta(win_length=3, input_shape=(4, 1, 1), data_format='channels_last'))

    delta_kapre = delta_model(specgrams)
    delta_ref = np.array([0.5, 1.0, 1.0, 0.5], dtype=np.float32)
    delta_ref = np.reshape(delta_ref, (1, -1, 1, 1))  # (b, t, f, ch)

    np.testing.assert_allclose(delta_kapre, delta_ref)


@pytest.mark.parametrize('data_format', ['default', 'channels_first', 'channels_last'])
def test_mag_phase(data_format):
    n_ch = 1
    n_fft, hop_length, win_length = 512, 256, 512

    src_mono, batch_src, input_shape = get_audio(data_format=data_format, n_ch=n_ch)

    mag_phase_layer = get_stft_mag_phase(
        input_shape=input_shape,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        input_data_format=data_format,
        output_data_format=data_format,
    )
    model = tensorflow.keras.models.Sequential()
    model.add(mag_phase_layer)
    mag_phase_kapre = model(batch_src)[0]  # a 2d image shape

    ch_axis = 0 if data_format == 'channels_first' else 2  # non-batch
    mag_phase_ref = np.stack(
        librosa.magphase(
            librosa.stft(
                src_mono,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                center=False,
            ).T
        ),
        axis=ch_axis,
    )
    np.testing.assert_equal(mag_phase_kapre.shape, mag_phase_ref.shape)
    # magnitude test
    np.testing.assert_allclose(
        np.take(
            mag_phase_kapre,
            [
                0,
            ],
            axis=ch_axis,
        ),
        np.take(
            mag_phase_ref,
            [
                0,
            ],
            axis=ch_axis,
        ),
        atol=2e-4,
    )
    # phase test - todo - yeah..


@pytest.mark.parametrize('waveform_data_format', ['default', 'channels_first', 'channels_last'])
@pytest.mark.parametrize('stft_data_format', ['default', 'channels_first', 'channels_last'])
@pytest.mark.parametrize('hop_ratio', [0.5, 0.25, 0.125])
def test_perfectly_reconstructing_stft_istft(waveform_data_format, stft_data_format, hop_ratio):
    n_ch = 1
    src_mono, batch_src, input_shape = get_audio(data_format=waveform_data_format, n_ch=n_ch)
    time_axis = 1 if waveform_data_format == 'channels_first' else 0  # non-batch!
    len_src = input_shape[time_axis]

    n_fft = 2048
    hop_length = int(2048 * hop_ratio)
    n_added_frames = int(1 / hop_ratio) - 1

    stft, istft = get_perfectly_reconstructing_stft_istft(
        stft_input_shape=input_shape,
        n_fft=n_fft,
        hop_length=hop_length,
        waveform_data_format=waveform_data_format,
        stft_data_format=stft_data_format,
    )
    # Test - [STFT -> ISTFT]
    model = tf.keras.models.Sequential([stft, istft])

    recon_waveform = model(batch_src)

    # trim off the pad_begin part
    len_pad_begin = n_fft - hop_length
    if waveform_data_format == 'channels_first':
        recon_waveform = recon_waveform[:, :, len_pad_begin : len_pad_begin + len_src]
    else:
        recon_waveform = recon_waveform[:, len_pad_begin : len_pad_begin + len_src, :]

    np.testing.assert_allclose(batch_src, recon_waveform, atol=1e-5)

    # Test - [ISTFT -> STFT]
    S = librosa.stft(src_mono, n_fft=n_fft, hop_length=hop_length).T.astype(
        np.complex64
    )  # (time, freq)

    ch_axis = 1 if stft_data_format == 'channels_first' else 3  # batch shape
    S = np.expand_dims(S, (0, ch_axis))
    model = tf.keras.models.Sequential([istft, stft])
    recon_S = model(S)

    # trim off the frames coming from zero-pad result
    n = n_added_frames
    n_added_frames += n
    if stft_data_format == 'channels_first':
        if n != 0:
            S = S[:, :, n:-n, :]
        recon_S = recon_S[:, :, n_added_frames:-n_added_frames, :]
    else:
        if n != 0:
            S = S[:, n:-n, :, :]
        recon_S = recon_S[:, n_added_frames:-n_added_frames, :, :]

    np.testing.assert_equal(S.shape, recon_S.shape)
    allclose_complex_numbers(S, recon_S)


@pytest.mark.parametrize('save_format', ['tf', 'h5'])
def test_save_load(save_format):
    """test saving/loading of models that has stft, melspectorgrma, and log frequency."""

    src_mono, batch_src, input_shape = get_audio(data_format='channels_last', n_ch=1)
    # test STFT save/load
    save_load_compare(
        STFT(input_shape=input_shape, pad_begin=True),
        batch_src,
        allclose_complex_numbers,
        save_format,
        STFT,
    )

    # test ConcatenateFrequencyMap
    specs_batch = np.random.randn(2, 3, 5, 4).astype(np.float32)
    save_load_compare(
        ConcatenateFrequencyMap(input_shape=specs_batch.shape[1:]),
        specs_batch,
        np.testing.assert_allclose,
        save_format,
        ConcatenateFrequencyMap,
    )

    if save_format == 'tf':
        # test melspectrogram save/load
        save_load_compare(
            get_melspectrogram_layer(input_shape=input_shape, return_decibel=True),
            batch_src,
            np.testing.assert_allclose,
            save_format,
        )
        # test log frequency spectrogram save/load
        save_load_compare(
            get_log_frequency_spectrogram_layer(input_shape=input_shape, return_decibel=True),
            batch_src,
            np.testing.assert_allclose,
            save_format,
        )
        # test stft_mag_phase
        save_load_compare(
            get_stft_mag_phase(input_shape=input_shape, return_decibel=True),
            batch_src,
            np.testing.assert_allclose,
            save_format,
        )
        # test stft mag
        save_load_compare(
            get_stft_magnitude_layer(input_shape=input_shape),
            batch_src,
            np.testing.assert_allclose,
            save_format,
        )


@pytest.mark.parametrize('data_format', ['default', 'channels_first', 'channels_last'])
def test_concatenate_frequency_map(data_format):
    shape = (4, 10, 5, 3)

    time_axis, freq_axis, ch_axis = (
        (1, 2, 3) if data_format == 'channels_last' else (2, 3, 1)
    )  # todo - replace it with CH_LAST_STR
    batch_size, n_freq, n_time, n_ch = shape[0], shape[freq_axis], shape[time_axis], shape[ch_axis]

    x = tf.random.normal(shape, dtype=tf.float32)
    concat_freq_map = ConcatenateFrequencyMap(data_format=data_format)
    x_concat = concat_freq_map(x).numpy()

    # test shape
    new_shape = list(shape)
    new_shape[ch_axis] += 1
    np.testing.assert_equal(tuple(new_shape), x_concat.shape)
    # test freq map
    freq_map = x_concat[0, 0, :, -1] if data_format == 'channels_last' else x_concat[0, -1, 0, :]
    np.testing.assert_allclose(freq_map, np.linspace(0, 1, num=n_freq))
    # test original input
    other_channels = (
        x_concat[:, :, :, :-1] if data_format == 'channels_last' else x_concat[:, :-1, :, :]
    )
    np.testing.assert_equal(x, other_channels)


@pytest.mark.parametrize('data_format', ['default', 'channels_first', 'channels_last'])
def test_get_frequency_aware_conv2d(data_format):
    shape = (4, 10, 5, 3)
    x = tf.random.normal(shape, dtype=tf.float32)

    freq_aware_conv2d = get_frequency_aware_conv2d(
        data_format, 'freq_aware_conv2d', 4, (3, 3), strides=(2, 2)
    )
    if (
        data_format != 'channels_first'
    ):  # because on cpu, channel_first conv doesn't work in typical tensorflow.
        _ = freq_aware_conv2d(x)


@pytest.mark.xfail()
@pytest.mark.parametrize('layer', [STFT, InverseSTFT])
def test_wrong_input_data_format(layer):
    _ = layer(input_data_format='weird_string')


@pytest.mark.xfail()
@pytest.mark.parametrize('layer', [STFT, InverseSTFT])
def test_wrong_input_data_format(layer):
    _ = layer(output_data_format='weird_string')


@pytest.mark.xfail()
@pytest.mark.parametrize('layer', [Delta, ApplyFilterbank])
def test_wrong_data_format(layer):
    _ = layer(data_format='weird_string')


if __name__ == '__main__':
    pytest.main([__file__])
