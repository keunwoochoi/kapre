import pytest
import numpy as np
from tensorflow.keras import backend as K
from kapre import backend as KPB
from kapre.backend import amplitude_to_decibel

TOL = 1e-5


def test_amplitude_to_decibel():
    """test for backend_keras.amplitude_to_decibel"""

    x = np.array([[1e-20, 1e-5, 1e-3, 5e-2], [0.3, 1.0, 20.5, 9999]])  # random positive numbers

    amin = 1e-5
    dynamic_range = 80.0

    x_decibel = 10 * np.log10(np.maximum(x, amin))
    x_decibel = x_decibel - np.max(x_decibel, axis=(1,), keepdims=True)
    x_decibel_ref = np.maximum(x_decibel, -1 * dynamic_range)

    x_var = K.variable(x)
    x_decibel_kapre = amplitude_to_decibel(x_var, amin, dynamic_range)

    assert np.allclose(K.eval(x_decibel_kapre), x_decibel_ref, atol=TOL)


@pytest.mark.parametrize('sample_rate', [44100, 22050])
@pytest.mark.parametrize('n_freq', [1025, 257])
@pytest.mark.parametrize('n_mels', [32, 128])
@pytest.mark.parametrize('f_min', [0.0, 200])
@pytest.mark.parametrize('f_max_ratio', [1.0, 0.5])
@pytest.mark.parametrize('htk', [True, False])
@pytest.mark.parametrize('norm', [None, 'slaney', 1.0])
def test_mel(sample_rate, n_freq, n_mels, f_min, f_max_ratio, htk, norm):
    """It only tests if the function is a valid wrapper"""
    f_max = int(f_max_ratio * (sample_rate // 2))
    mel_fb = KPB.filterbank_mel(sample_rate=sample_rate,
                                n_freq=n_freq,
                                n_mels=n_mels,
                                f_min=f_min,
                                f_max=f_max,
                                htk=htk,
                                norm=norm)

    assert mel_fb.dtype == K.floatx()
    assert mel_fb.shape == (n_freq, n_mels)


@pytest.mark.parametrize('sample_rate', [44100, 22050])
@pytest.mark.parametrize('n_freq', [1025, 257])
@pytest.mark.parametrize('n_bins', [32, 84])
@pytest.mark.parametrize('bins_per_octave', [8, 12, 36])
@pytest.mark.parametrize('f_min', [1.0, 0.5])
@pytest.mark.parametrize('spread', [0.5, 0.125])
def test_filterbank_log(sample_rate, n_freq, n_bins, bins_per_octave, f_min, spread):
    """It only tests if the function is a valid wrapper"""
    log_fb = KPB.filterbank_log(sample_rate=sample_rate,
                                n_freq=n_freq,
                                n_bins=n_bins,
                                bins_per_octave=bins_per_octave,
                                f_min=f_min,
                                spread=spread)

    assert log_fb.dtype == K.floatx()
    assert log_fb.shape == (n_freq, n_bins)


@pytest.mark.xfail()
def test_fb_log_fail():
    _ = KPB.filterbank_log(sample_rate=22050,
                           n_freq=513,
                           n_bins=300,
                           bins_per_octave=12)


if __name__ == '__main__':
    pytest.main([__file__])
