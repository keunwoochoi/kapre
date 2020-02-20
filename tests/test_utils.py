import pytest
import numpy as np
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.backend import image_data_format
import kapre
import librosa
from kapre.utils import AmplitudeToDB, Normalization2D

TOL = 1e-5


def test_amplitude_to_db():
    """test for AmplitudeToDB layer"""

    # Test for a normal case
    model = tensorflow.keras.models.Sequential()
    model.add(AmplitudeToDB(amin=1e-10, top_db=80.0,
                            input_shape=(6,)))

    x = np.array([0, 1e-5, 1e-3, 1e-2, 1e-1, 1])
    x_db_ref = np.array([-80, -50, -30, -20, -10, 0])
    batch_x_db = model.predict(x[np.newaxis, :])
    assert np.allclose(batch_x_db[0], x_db_ref, atol=TOL)

    # Smaller amin, bigger dynamic range
    model = tensorflow.keras.models.Sequential()
    model.add(AmplitudeToDB(amin=1e-12, top_db=120.0,
                            input_shape=(6,)))
    x = np.array([1e-15, 1e-10, 1e-5, 1e-2, 1e-1, 10])
    x_db_ref = np.array([-120, -110, -60, -30, -20, 0])
    batch_x_db = model.predict(x[np.newaxis, :])
    assert np.allclose(batch_x_db[0], x_db_ref, atol=TOL)

    # TODO: Saving and loading the model


def test_normalization_2d():
    """test for Normalization2D"""
    # TODO: Because the expected behaviour of this layer is somehow confusing for me now


if __name__ == '__main__':
    pytest.main([__file__])
