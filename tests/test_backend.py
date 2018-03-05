import os
import pytest
import kapre
from kapre import backend as KPB
from keras import backend as K
import numpy as np

TOL = 1e-5


def test_amplitude_to_decibel():
    """test for backend_keras.amplitude_to_decibel"""
    from kapre.backend_keras import amplitude_to_decibel
    x = np.array([1e-20, 1e-5, 1e-3, 5e-2, 0.3, 1.0, 20.5, 9999])  # random positive numbers

    amin = 1e-5
    dynamic_range = 80.0

    x_decibel = 10 * np.log10(np.maximum(x, amin))
    x_decibel = x_decibel - np.max(x_decibel)
    x_decibel_ref = np.maximum(x_decibel, -1 * dynamic_range)

    x_var = K.variable(x)
    x_decibel_kapre = amplitude_to_decibel(x_var, amin, dynamic_range)

    assert np.allclose(K.eval(x_decibel_kapre), x_decibel_ref, atol=TOL)


def test_mel():
    """test for backend.mel_frequencies
    For librosa wrappers, it only tests the data type of returned value
    """
    assert KPB.mel(sr=22050, n_dft=512).dtype == K.floatx()


def test_get_stft_kernels():
    """test for backend.get_stft_kernels"""
    n_dft = 4
    real_kernels, imag_kernels = KPB.get_stft_kernels(n_dft)

    real_kernels_ref = np.array([[[[0.0, 0.0, 0.0]]],
                                 [[[0.5, 0.0, -0.5]]],
                                 [[[1.0, -1.0, 1.0]]],
                                 [[[0.5, 0.0, -0.5]]]], dtype=K.floatx())
    imag_kernels_ref = np.array([[[[0.0, 0.0, 0.0]]],
                                 [[[0.0, -0.5, 0.0]]],
                                 [[[0.0, 0.0, 0.0]]],
                                 [[[0.0, 0.5, 0.0]]]], dtype=K.floatx())

    assert real_kernels.shape == (n_dft, 1, 1, n_dft // 2 + 1)
    assert imag_kernels.shape == (n_dft, 1, 1, n_dft // 2 + 1)
    assert np.allclose(real_kernels, real_kernels_ref, atol=TOL)
    assert np.allclose(imag_kernels, imag_kernels_ref, atol=TOL)


def test_filterbank_mel():
    """test for backend.filterbank_mel"""
    fbmel_ref = np.load(os.path.join(os.path.dirname(__file__), 'fbmel_8000_512.npy'))
    fbmel = KPB.filterbank_mel(sr=8000, n_freq=512)
    assert fbmel.shape == fbmel_ref.shape
    assert np.allclose(fbmel, fbmel_ref, atol=TOL)


def test_filterbank_log():
    """test for backend.filterback_log"""
    fblog_ref = np.load(os.path.join(os.path.dirname(__file__), 'fblog_8000_512.npy'))
    fblog = KPB.filterbank_log(sr=8000, n_freq=512)
    assert fblog.shape == fblog_ref.shape
    assert np.allclose(fblog, fblog_ref, atol=TOL)


if __name__ == '__main__':
    pytest.main([__file__])
