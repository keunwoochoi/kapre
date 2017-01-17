# kapre
Keras Audio Preprocessors

## Installation
```bash
$ pip install kapre
```

## Status
* In [`dev` branch](https://github.com/keunwoochoi/kapre/tree/dev), a FFT-based `Stft` layer is added.
* In [`dev-aug` branch](https://github.com/keunwoochoi/kapre/tree/dev-aug), a additive white noise layer is added.

They will be merged into master at some point.

# Layers

* `Spectrogram`, `Melspectrogram` in [time_frequency.py](https://github.com/keunwoochoi/kapre/blob/master/kapre/time_frequency.py)
* `Filterbank` in [filterbank.py](https://github.com/keunwoochoi/kapre/blob/master/kapre/time_frequency.py)
* `AmplitudeToDB`, `Normalization2D` in [utils.py](https://github.com/keunwoochoi/kapre/blob/master/kapre/utils.py)


## Usage

### Spectrogram
```python
from keras.models import Sequential
from kapre.time_frequency import Spectrogram
from kapre.utils import Normalization2D

# input: stereo channel, 1-sec audio signal
input_shape = (2, 44100) 
sr = 44100
model = Sequential()
# A "power-spectrogram in decibel scale" layer
model.add(Spectrogram(n_dft=512, n_hop=256, input_shape=src_shape,
                      border_mode='same', power_spectrogram=2.0,
                      return_decibel=True, trainable_kernel=False,
                      name='trainable_stft'))
# If you wanna normalise it per-frequency
model.add(Normalization2D(str_axis='freq')) # or 'time', 'channel', 'batch', 'data_sample'
# Then add your model
# E.g., model.add(some convolution layers...)
```

### Mel-spectrogram
```python
from keras.models import Sequential
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D

# 6 channels (!), maybe 1-sec audio signal
input_shape = (6, 44100) 
sr = 44100
model = Sequential()
# A mel-spectrogram layer with
# no decibel conversion for some reasons and (return_decibel=False)
# amplitude, not power (power=1.0)
model.add(Melspectrogram(n_dft=512, n_hop=256, input_shape=src_shape,
                         border_mode='same', sr=sr, n_mels=128,
                         fmin=0.0, fmax=sr/2, power=1.0,
                         return_decibel=False, trainable_fb=False,
                         trainable_kernel=False
                         name='trainable_stft'))
# If you wanna normalise it per-channel
model.add(Normalization2D(str_axis='channel')) # or 'freq', 'time', 'batch', 'data_sample'
# Then add your model
# E.g., model.add(some convolution layers...)
```

## When you wanna save/load model w these layers

Use `custom_objects` keyword argument as below.

```python
import keras
import kapre

model = keras.models.Sequential()
model.add(kapre.time_frequency.Melspectrogram(512, input_shape=(1, 44100)))
model.summary()
model.save('temp_model.h5')

model2 = keras.models.load_model('temp_model.h5', 
  custom_objects={'Melspectrogram':kapre.TimeFrequency.Melspectrogram})
model2.summary()
```

# Documentation
Please read docstrings at this moment.

# Plan

  - [x] `time_frequency`: Spectrogram, Mel-spectrogram
  - [x] `utils`: AmplitudeToDB, Normalization2D, A-weighting
  - [x] `filterbank`: filterbanks
  - [ ] `time_frequency`: FFT-based STFT (developing...)
  - [ ] `data_augmentation`: Random-gain Gaussian noise (developing...)
  - [ ] `data_augmentation`: random cropping 1D/2D, Dynamic Range Compression1D
  - [ ] `utils`: A-weighting
  - [ ] `filterbank`: Parameteric Filter bank
  - [ ] `Decompose`: Harmonic-Percussive separation
  - [ ] `InverseSpectrogram`
  - [ ] `TimeFrequency`: Harmonic/Spiral representations, chromagram

# Citation
Citation is required if you used `kapre` in your paper.

```
@article{choi2016kapre,
  title={kapre: Keras Audio PREprocessors},
  author={Choi, Keunwoo},
  journal={GitHub repository: https://github.com/keunwoochoi/kapre},
  year={2016}
}
```