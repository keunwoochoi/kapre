import pytest
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import backend as K
from kapre import backend as KPB
from kapre.backend import magnitude_to_decibel, validate_data_format_str

from utils import SRC

TOL = 1e-5


@pytest.mark.parametrize('dynamic_range', [80.0, 120.0])
@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64'])
def test_magnitude_to_decibel(dynamic_range, dtype: str):
    """test for backend_keras.magnitude_to_decibel"""

    x = np.array(
        [[1e-20, 1e-5, 1e-3, 5e-2], [0.3, 1.0, 20.5, 9999]], dtype=dtype
    )  # random positive numbers

    amin = 1e-5
    x_decibel_ref = np.stack(
        (
            librosa.power_to_db(x[0], amin=amin, ref=1.0, top_db=dynamic_range),
            librosa.power_to_db(x[1], amin=amin, ref=1.0, top_db=dynamic_range),
        ),
        axis=0,
    )

    x_var = K.variable(x)
    x_decibel_kapre = magnitude_to_decibel(
        x_var, ref_value=1.0, amin=amin, dynamic_range=dynamic_range
    )
    if dtype == 'float16':
        np.testing.assert_allclose(K.eval(x_decibel_kapre), x_decibel_ref, rtol=1e-3, atol=TOL)
    else:
        np.testing.assert_allclose(K.eval(x_decibel_kapre), x_decibel_ref, atol=TOL)


@pytest.mark.parametrize('sample_rate', [44100, 22050])
@pytest.mark.parametrize('n_freq', [1025, 257])
@pytest.mark.parametrize('n_mels', [32, 128])
@pytest.mark.parametrize('f_min', [0.0, 200])
@pytest.mark.parametrize('f_max_ratio', [1.0, 0.5])
@pytest.mark.parametrize('htk', [True, False])
@pytest.mark.parametrize('norm', [None, 'slaney', 1.0])
def test_mel(sample_rate, n_freq, n_mels, f_min, f_max_ratio, htk, norm):
    f_max = int(f_max_ratio * (sample_rate // 2))
    mel_fb = KPB.filterbank_mel(
        sample_rate=sample_rate,
        n_freq=n_freq,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        htk=htk,
        norm=norm,
    )
    mel_fb = mel_fb.numpy()

    mel_fb_ref = librosa.filters.mel(
        sr=sample_rate,
        n_fft=(n_freq - 1) * 2,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        htk=htk,
        norm=norm,
    ).T

    assert mel_fb.dtype == K.floatx()
    assert mel_fb.shape == (n_freq, n_mels)
    np.testing.assert_allclose(mel_fb_ref, mel_fb)


@pytest.mark.parametrize('sample_rate', [44100, 22050])
@pytest.mark.parametrize('n_freq', [1025, 257])
@pytest.mark.parametrize('n_bins', [32, 84])
@pytest.mark.parametrize('bins_per_octave', [8, 12, 36])
@pytest.mark.parametrize('f_min', [1.0, 0.5])
@pytest.mark.parametrize('spread', [0.5, 0.125])
def test_filterbank_log(sample_rate, n_freq, n_bins, bins_per_octave, f_min, spread):
    """It only tests if the function is a valid wrapper"""
    log_fb = KPB.filterbank_log(
        sample_rate=sample_rate,
        n_freq=n_freq,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        f_min=f_min,
        spread=spread,
    )

    assert log_fb.dtype == K.floatx()
    assert log_fb.shape == (n_freq, n_bins)


@pytest.mark.parametrize('quantization_channels', [100, 256])
def test_mu_law_correctness(quantization_channels):
    # test reconstruction
    mu_src = np.arange(0, quantization_channels).astype(np.int)
    src = KPB.mu_law_decoding(mu_src, quantization_channels=quantization_channels)
    mu_src_recon = KPB.mu_law_encoding(src, quantization_channels=quantization_channels)

    np.testing.assert_equal(mu_src, mu_src_recon)

    # test against librosa
    resol = 1 / (2 ** 16)
    src = np.arange(-1.0, 1.0, resol).astype(np.float32)
    mu = quantization_channels - 1
    mu_src_ref = librosa.mu_compress(src, mu=quantization_channels - 1, quantize=False)
    mu_src_ref = (mu_src_ref + 1.0) / 2.0 * mu + 0.5
    mu_src_ref = mu_src_ref.astype(np.int)

    mu_src_kapre = KPB.mu_law_encoding(
        tf.convert_to_tensor(src), quantization_channels=quantization_channels
    )
    np.testing.assert_equal(mu_src_ref, mu_src_kapre.numpy())


@pytest.mark.xfail()
def test_fb_log_fail():
    _ = KPB.filterbank_log(sample_rate=22050, n_freq=513, n_bins=300, bins_per_octave=12)


@pytest.mark.xfail()
def test_unsupported_window():
    _ = KPB.get_window_fn('wrong_window_name')


@pytest.mark.xfail()
def test_validate_fail():
    _ = validate_data_format_str('weird_string')


if __name__ == '__main__':
    pytest.main([__file__])
