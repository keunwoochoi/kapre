__version__ = '0.3.7'
VERSION = __version__

from . import composed
from . import backend

# Signal processing layers
from .signal import Frame, Energy, MuLawEncoding, MuLawDecoding, LogmelToMFCC

# Time-frequency layers
from .time_frequency import (
    STFT,
    InverseSTFT,
    Magnitude,
    Phase,
    MagnitudeToDecibel,
    ApplyFilterbank,
    Delta,
    ConcatenateFrequencyMap,
)

# TFLite compatible layers
from .time_frequency_tflite import STFTTflite, MagnitudeTflite, PhaseTflite

# Augmentation layers
from .augmentation import SpecAugment, ChannelSwap

# Composed layers (higher-level functions)
from .composed import (
    get_stft_magnitude_layer,
    get_melspectrogram_layer,
    get_log_frequency_spectrogram_layer,
    get_perfectly_reconstructing_stft_istft,
    get_stft_mag_phase,
    get_frequency_aware_conv2d,
)

__all__ = [
    # Version info
    '__version__',
    'VERSION',

    # Signal processing layers
    'Frame',
    'Energy',
    'MuLawEncoding',
    'MuLawDecoding',
    'LogmelToMFCC',

    # Time-frequency layers
    'STFT',
    'InverseSTFT',
    'Magnitude',
    'Phase',
    'MagnitudeToDecibel',
    'ApplyFilterbank',
    'Delta',
    'ConcatenateFrequencyMap',

    # TFLite compatible layers
    'STFTTflite',
    'MagnitudeTflite',
    'PhaseTflite',

    # Augmentation layers
    'SpecAugment',
    'ChannelSwap',

    # Composed layers (higher-level functions)
    'get_stft_magnitude_layer',
    'get_melspectrogram_layer',
    'get_log_frequency_spectrogram_layer',
    'get_perfectly_reconstructing_stft_istft',
    'get_stft_mag_phase',
    'get_frequency_aware_conv2d',
]
