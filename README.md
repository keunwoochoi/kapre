# kapre
Keras Audio Preprocessors

## News
* 2x June 2017
  - Kapre ver 0.1
    - Remove STFT, python3 compatible
    - A full documentation in this readme.md
    - pip version is updated
* 15 May 2017
  - Jamendo dataset loader is added.
* 18 March 2017
  - [`dataset.py`](https://github.com/keunwoochoi/kapre/blob/master/kapre/datasets.py); GTZan, MagnaTagATune, MusicNet, FMA are available.

* 16 March 2017 (kapre v0.0.3.1)
  - Compatible to Keras 2.0. Kapre won't support Keras 1.0 and require Keras 2.0 now.
  - There's no change on Kapre API and you can just use, save, and load.
  - Stft is not working and will be fixed later.

* 15 March 2017 (kapre v0.0.3)
  - [`dataset.py`](https://github.com/keunwoochoi/kapre/blob/master/kapre/datasets.py) is added.

## Installation

1. For keras >= 2.0
```
$ pip install kapre
```
Or,
```
$ git clone https://github.com/keunwoochoi/kapre.git
$ cd kapre
$ python setup.py install
```

2. For Keras 1.x (note: it is not up-to-date)
```
$ git clone https://github.com/keunwoochoi/kapre.git
$ cd kapre
$ python setup.py install
$ cd kapre
$ git checkout a2bde3e
$ python setup.py install
```

## Layers
* `Spectrogram`, `Melspectrogram` in [time_frequency.py](https://github.com/keunwoochoi/kapre/blob/master/kapre/time_frequency.py)
* `AmplitudeToDB`, `Normalization2D` in [utils.py](https://github.com/keunwoochoi/kapre/blob/master/kapre/utils.py)
* `Filterbank` in [filterbank.py](https://github.com/keunwoochoi/kapre/blob/master/kapre/time_frequency.py)
* `AdditiveNoise` in [augmentation.py](https://github.com/keunwoochoi/kapre/blob/master/kapre/augmentation.py)

## Datasets
* [GTZan](http://marsyasweb.appspot.com/download/data_sets/): (30s, 10 genres, 1,000 mp3)
* [MagnaTagATune](http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset): (29s, 188 tags, 25,880 mp3) for tagging and triplet similarity
* [MusicNet](https://homes.cs.washington.edu/~thickstn/musicnet.html): (full length 330 classicals music, note-wise annotations)
* [FMA](https://github.com/mdeff/fma): small/medium/large/full collections, up to 100+K songs from free music archieve, for genre classification. With genre hierarchy, pre-computed features, splits, etc.
* [Jamendo](http://www.mathieuramona.com/wp/data/jamendo/): 61/16/24 songs for vocal activity detection

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
                         return_decibel_melgram=False, trainable_fb=False,
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
kapre.datasets.load_fma('datasets', size='small')
kapre.datasets.load_fma('datasets', size='medium')
kapre.datasets.load_musicnet('datasets', format='hdf')
kapre.datasets.load_musicnet('datasets', format='npz')
# Kapre does NOT remove zip/tar.gz files after extracting.
```

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

# Documentation
## `time_frequency`
### `Spectrogram`

`kapre.time_frequency.spectrogram`

Spectrogram layer that outputs spectrogram(s) in 2D image format.

#### Parameters
 * n_dft: int > 0 [scalar]
   - The number of DFT points, presumably power of 2.
   - Default: ``512``

 * n_hop: int > 0 [scalar]
   - Hop length between frames in sample,  probably <= ``n_dft``.
   - Default: ``None`` (``n_dft / 2`` is used)

 * padding: str, ``'same'`` or ``'valid'``.
   - Padding strategies at the ends of signal.
   - Default: ``'same'``

 * power_spectrogram: float [scalar],
   - ``2.0`` to get power-spectrogram, ``1.0`` to get amplitude-spectrogram.
   - Usually ``1.0`` or ``2.0``.
   - Default: ``2.0``

 * return_decibel_spectrogram: bool,
    - Whether to return in decibel or not, i.e. returns log10(amplitude spectrogram) if ``True``.
    - Recommended to use ``True``, although it's not by default.
    - Default: ``False``

 * trainable_kernel: bool
   -  Whether the kernels are trainable or not.
   -  If ``True``, Kernels are initialised with DFT kernels and then trained.
   -  Default: ``False``

* image_data_format: string, ``'channels_first'`` or ``'channels_last'``.
   -  The returned spectrogram follows this image_data_format strategy.
   -  If ``'default'``, it follows the current Keras session's setting.
   -  Setting is in ``./keras/keras.json``.
   -  Default: ``'default'``

#### Notes
 * The input should be a 2D array, ``(audio_channel, audio_length)``.
 * E.g., ``(1, 44100)`` for mono signal, ``(2, 44100)`` for stereo signal.
 * It supports multichannel signal input, so ``audio_channel`` can be any positive integer.

#### Returns

A Keras layer

 * abs(Spectrogram) in a shape of 2D data, i.e.,
 * `(None, n_channel, n_freq, n_time)` if `'channels_first'`,
 * `(None, n_freq, n_time, n_channel)` if `'channels_last'`,




