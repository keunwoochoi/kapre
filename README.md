# kapre
Keras Audio Preprocessors

## News
* 15 March 2017 - [`dataset.py`](https://github.com/keunwoochoi/kapre/blob/master/kapre/datasets.py) is added.

## Installation
```
$ git clone https://github.com/keunwoochoi/kapre.git
$ cd kapre
$ python setup.py install
```
(Kapre is on pip, but pip version is not always up-to-date. 
So please use git version until it becomes more stable.)

## Layers

* `Spectrogram`, `Melspectrogram` in [time_frequency.py](https://github.com/keunwoochoi/kapre/blob/master/kapre/time_frequency.py)
* `Stft` in [stft.py](https://github.com/keunwoochoi/kapre/blob/master/kapre/stft.py)
* `Filterbank` in [filterbank.py](https://github.com/keunwoochoi/kapre/blob/master/kapre/time_frequency.py)
* `AmplitudeToDB`, `Normalization2D` in [utils.py](https://github.com/keunwoochoi/kapre/blob/master/kapre/utils.py)
* `AdditiveNoise` in [augmentation.py](https://github.com/keunwoochoi/kapre/blob/master/kapre/augmentation.py)

## Usage Example

* For real, working code: checkout [example folder](https://github.com/keunwoochoi/kapre/tree/master/examples)

### Mel-spectrogram
```python
from keras.models import Sequential
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
from kapre.augmentation import AdditiveNoise

# 6 channels (!), maybe 1-sec audio signal
input_shape = (6, 44100) 
sr = 44100
model = Sequential()
# A mel-spectrogram layer
model.add(Melspectrogram(n_dft=512, n_hop=256, input_shape=input_shape,
                         border_mode='same', sr=sr, n_mels=128,
                         fmin=0.0, fmax=sr/2, power=1.0,
                         return_decibel=False, trainable_fb=False,
                         trainable_kernel=False,
                         name='trainable_stft'))
# Maybe some additive white noise.
model.add(AdditiveNoise(power=0.2))
# If you wanna normalise it per-frequency
model.add(Normalization2D(str_axis='freq')) # or 'channel', 'time', 'batch', 'data_sample'
# After this, it's just a usual keras workflow. For example..
# Add some layers, e.g., model.add(some convolution layers..)
# Compile the model
model.compile('adam', 'categorical_crossentropy') # if single-label classification
# train it with raw audio sample inputs
x = load_x() # e.g., x.shape = (10000, 6, 44100)
y = load_y() # e.g., y.shape = (10000, 10) if it's 10-class classification
# and train it
model.fit(x, y)
# write a paper and graduate or get paid. Profit!
```

###  When you wanna save/load model w these layers

Use `custom_objects` keyword argument as below.

```python
import keras
import kapre

model = keras.models.Sequential()
model.add(kapre.time_frequency.Melspectrogram(512, input_shape=(1, 44100)))
model.summary()
model.save('temp_model.h5')

model2 = keras.models.load_model('temp_model.h5', 
  custom_objects={'Melspectrogram':kapre.time_frequency.Melspectrogram})
model2.summary()
```

### Downloading datasets
```python
import kapre

kapre.datasets.load_gtzan_genre('datasets')
# checkout datasets/gtzan,
# also `datasets/gtzan_genre/dataset_summary_kapre.csv`
kapre.datasets.load_magnatagatune('/Users/username/all_datasets')
# for magnatagatune, it doesn't create csv file as it already come with.
kapre.datasets.load_gtzan_speechmusic('datasets')
# check out `datasets/gtzan_speechmusic/dataset_summary_kapre.csv`

```

# Documentation
Please read docstrings at this moment. Fortunately I quite enjoy writing docstrings.

# Plan

  - [x] `time_frequency`: Spectrogram, Mel-spectrogram
  - [x] `utils`: AmplitudeToDB, Normalization2D
  - [x] `filterbank`: filterbanks (init with mel)
  - [x] `stft`: FFT-based STFT (Done for theano-backend only)
  - [x] `data_augmentation`: (Random-gain) white noise
  - [x] `datasets.py`: download and manage MIR datasets.
  - [ ] `data_augmentation`: Dynamic Range Compression1D, some other noises
  - [ ] `utils`: A-weighting
  - [ ] `filterbank`: Parameteric Filter bank
  - [ ] `Decompose`: Harmonic-Percussive separation
  - [ ] `InverseSpectrogram`: istft, (+ util: magnitude + phase merger)
  - [ ] `TimeFrequency`: Harmonic/Spiral representations, chromagram, tempogram

# Citation
Please cite it as...

```
@article{choi2016kapre,
  title={kapre: Keras Audio PREprocessors},
  author={Choi, Keunwoo},
  journal={GitHub repository: https://github.com/keunwoochoi/kapre},
  year={2016}
}
```