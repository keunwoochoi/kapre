# kapre
Keras Audio Preprocessors. Written by Keunwoo Choi.

Why bother to save STFT/melspectrograms to your storage? Just do it on-the-fly on-GPU.

How demanding is the computation? [Check out this paper!](https://arxiv.org/abs/1706.05781)

## Contents
- [News](#news)
- [Installation](#installation)
- [Usage](#usage)
- [One-shot example](#one-shot-example)
- [How to cite](#citation)
- [API Documentation](#api-documentation)
  - [`time_frequency.Spectrogram`](#spectrogram)
  - [`time_frequency.Melspectrogram`](#melspectrogram)
  - [`utils.AmplitudeToDB`](#amplitudetodb)
  - [`utils.Normalization2D`](#normalization2d)
  - [`filterbank.Filterbank`](#filterbank)
  - [`augmentation.AdditiveNoise`](#additivenoise)

## News
* 20 Feb 2019
  - Kapre ver 0.1.4
    - Fixed amplitude-to-decibel error as raised in [#46](https://github.com/keunwoochoi/kapre/issues/46)
     
* March 2018
  - Kapre ver 0.1.3
    - Kapre is on Pip again
    - Add unit tests
    - Remove `Datasets`
    - Remove some codes while adding more dependency on Librosa to make it cleaner and more stable
      - and therefore `htk` option enabled in `Melspectrogram`
* 9 July 2017
  - Kapre ver 0.1.1, aka 'pretty stable' with a [benchmark paper](https://arxiv.org/abs/1706.05781)
    - Remove STFT, python3 compatible
    - A full documentation in this readme.md
    - pip version is updated

## Installation
([↑up to contents](#contents))

1. For keras >= 2.0
```sh
pip install kapre
```
Or,
```sh
git clone https://github.com/keunwoochoi/kapre.git
cd kapre
python setup.py install
```

2. For Keras 1.x (note: it is not updated anymore;)
```sh
git clone https://github.com/keunwoochoi/kapre.git
cd kapre
git checkout a2bde3e
python setup.py install
```

## Usage
### Layers
([↑up to contents](#contents))

Audio preprocessing layers
* `Spectrogram`, `Melspectrogram` in [time_frequency.py](https://github.com/keunwoochoi/kapre/blob/master/kapre/time_frequency.py)
* `AmplitudeToDB`, `Normalization2D` in [utils.py](https://github.com/keunwoochoi/kapre/blob/master/kapre/utils.py)
* `Filterbank` in [filterbank.py](https://github.com/keunwoochoi/kapre/blob/master/kapre/time_frequency.py)
* `AdditiveNoise` in [augmentation.py](https://github.com/keunwoochoi/kapre/blob/master/kapre/augmentation.py)

## One-shot example
([↑up to contents](#contents))

* More examples on [example folder](https://github.com/keunwoochoi/kapre/tree/master/examples)

### Using Mel-spectrogram
```python
from keras.models import Sequential
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
from kapre.augmentation import AdditiveNoise

# 6 channels (!), maybe 1-sec audio signal, for an example.
input_shape = (6, 44100)
sr = 44100
model = Sequential()
# A mel-spectrogram layer
model.add(Melspectrogram(n_dft=512, n_hop=256, input_shape=input_shape,
                         padding='same', sr=sr, n_mels=128,
                         fmin=0.0, fmax=sr/2, power_melgram=1.0,
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
# Done!
```

### To save/load models with kapre layers

Use `custom_objects` keyword argument as below.

```python
import keras
import kapre

model = keras.models.Sequential()
model.add(kapre.time_frequency.Melspectrogram(512, input_shape=(1, 44100)))
model.summary()
model.save('temp_model.h5')
# Now saved, let's load it.
model2 = keras.models.load_model('temp_model.h5',
  custom_objects={'Melspectrogram':kapre.time_frequency.Melspectrogram})
model2.summary()
```


# Citation
([↑up to contents](#contents))

Please cite this paper if you use Kapre for your work.

```
@inproceedings{choi2017kapre,
  title={Kapre: On-GPU Audio Preprocessing Layers for a Quick Implementation of Deep Neural Network Models with Keras},
  author={Choi, Keunwoo and Joo, Deokjin and Kim, Juho},
  booktitle={Machine Learning for Music Discovery Workshop at 34th International Conference on Machine Learning},
  year={2017},
  organization={ICML}
}
```

# API Documentation
## [time_frequency.py](kapre/time_frequency.py)
### Spectrogram
([↑up to contents](#contents))

```python
kapre.time_frequency.Spectrogram(n_dft=512, n_hop=None, padding='same',
                                 power_spectrogram=2.0, return_decibel_spectrogram=False,
                                 trainable_kernel=False, image_data_format='default',
                                 **kwargs)
```
Spectrogram layer that outputs spectrogram(s) in 2D image format.

#### Parameters

     * n_dft: int > 0 [scalar]
       - The number of DFT points, presumably power of 2.
       - Default: `512`
     * n_hop: int > 0 [scalar]
       - Hop length between frames in sample,  probably <= `n_dft`.
       - Default: `None` (`n_dft / 2` is used)
     * padding: str, `'same'` or `'valid'`.
       - Padding strategies at the ends of signal.
       - Default: `'same'`
     * power_spectrogram: float [scalar],
       - `2.0` to get power-spectrogram, `1.0` to get amplitude-spectrogram.
       - Usually `1.0` or `2.0`.
       - Default: `2.0`
     * return_decibel_spectrogram: bool,
        - Whether to return in decibel or not, i.e. returns log10(amplitude spectrogram) if `True`.
        - Recommended to use `True`, although it's not by default.
        - Default: `False`
     * trainable_kernel: bool
       -  Whether the kernels are trainable or not.
       -  If `True`, Kernels are initialised with DFT kernels and then trained.
       -  Default: `False`
    * image_data_format: string, `'channels_first'` or `'channels_last'`.
       -  The returned spectrogram follows this image_data_format strategy.
       -  If `'default'`, it follows the current Keras session's setting.
       -  Setting is in `./keras/keras.json`.
       -  Default: `'default'`

#### Notes

     * The input should be a 2D array, `(audio_channel, audio_length)`.
     * E.g., `(1, 44100)` for mono signal, `(2, 44100)` for stereo signal.
     * It supports multichannel signal input, so `audio_channel` can be any positive integer.

#### Returns

A Keras layer.

     * abs(Spectrogram) in a shape of 2D data, i.e.,
     * `(None, n_channel, n_freq, n_time)` if `'channels_first'`,
     * `(None, n_freq, n_time, n_channel)` if `'channels_last'`,

### Melspectrogram
([↑up to contents](#contents))

```python
kapre.time_frequency.Melspectrogram(sr=22050, n_mels=128, fmin=0.0, fmax=None,
                                    power_melgram=1.0, return_decibel_melgram=False,
                                    trainable_fb=False, **kwargs)
```
Mel-spectrogram layer that outputs mel-spectrogram(s) in 2D image format.

Its base class is `Spectrogram`.

Mel-spectrogram is an efficient representation using the property of human
auditory system -- by compressing frequency axis into mel-scale axis.

#### Parameters

     * sr: integer > 0 [scalar]
       - sampling rate of the input audio signal.
       - Default: `22050`
     * n_mels: int > 0 [scalar]
       - The number of mel bands.
       - Default: `128`
     * fmin: float > 0 [scalar]
       - Minimum frequency to include in Mel-spectrogram.
       - Default: `0.0`
     * fmax: float > `fmin` [scalar]
       - Maximum frequency to include in Mel-spectrogram.
       - If `None`, it is inferred as `sr / 2`.
       - Default: `None`
     * power_melgram: float [scalar]
       - Power of `2.0` if power-spectrogram,
       - `1.0` if amplitude spectrogram.
       - Default: `1.0`
     * return_decibel_melgram: bool
       - Whether to return in decibel or not, i.e. returns log10(amplitude spectrogram) if `True`.
       - Recommended to use `True`, although it's not by default.
       - Default: `False`
     * trainable_fb: bool
       - Whether the spectrogram -> mel-spectrogram filterbanks are trainable.
       - If `True`, the frequency-to-mel matrix is initialised with mel frequencies but trainable.
       - If `False`, it is initialised and then frozen.
       - Default: `False`
     * **kwargs:
       - The keyword arguments of `Spectrogram` such as `n_dft`, `n_hop`,
       - `padding`, `trainable_kernel`, `image_data_format`.

#### Notes

     * The input should be a 2D array, `(audio_channel, audio_length)`.
    E.g., `(1, 44100)` for mono signal, `(2, 44100)` for stereo signal.
     * It supports multichannel signal input, so `audio_channel` can be any positive integer.

#### Returns

A Keras layer

     * abs(mel-spectrogram) in a shape of 2D data, i.e.,
     * `(None, n_channel, n_mels, n_time)` if `'channels_first'`,
     * `(None, n_mels, n_time, n_channel)` if `'channels_last'`,



## [utils.py](kapre/utils.py)
### AmplitudeToDB
([↑up to contents](#contents))

```python
kapre.utils.AmplitudeToDB(amin=1e-10, top_db=80.0, **kwargs)
```

A layer that converts amplitude to decibel

#### Parameters

    * amin: float [scalar]
        - Noise floor. Default: 1e-10
    * top_db: float [scalar]
        - Dynamic range of output. Default: 80.0

#### Example
Adding `AmplitudeToDB` after a spectrogram:
```python
model.add(Spectrogram(return_decibel=False))
model.add(AmplitudeToDB())
```
, which is the same as:
```python
model.add(Spectrogram(return_decibel=True))
```

### Normalization2D
([↑up to contents](#contents))

```python
kapre.utils.Normalization2D(str_axis=None, int_axis=None, image_data_format='default',
                            eps=1e-10, **kwargs)
```

A layer that normalises input data in `axis` axis.

#### Parameters

    * input_shape: tuple of ints
        - E.g., `(None, n_ch, n_row, n_col)` if theano.
    * str_axis: str
        - used ONLY IF `int_axis` is `None`.
        - `'batch'`, `'data_sample'`, `'channel'`, `'freq'`, `'time')`
        - Even though it is optional, actually it is recommended to use
        - `str_axis` over `int_axis` because it provides more meaningful
        - and image data format-robust interface.
    * int_axis: int
        - axis index that along which mean/std is computed.
        - `0` for per data sample, `-1` for per batch.
        - `1`, `2`, `3` for channel, row, col (if channels_first)
        - if `int_axis is None`, `str_axis` SHOULD BE set.

#### Example

A frequency-axis normalization after a spectrogram::
    ```python
    model.add(Spectrogram())
    model.add(Normalization2D(str_axis='freq'))
    ```

## [filterbank.py](kapre/filterbank.py)
### Filterbank
([↑up to contents](#contents))

```python
kapre.filterbank.Filterbank(n_fbs, trainable_fb, sr=None, init='mel', fmin=0., fmax=None,
                            bins_per_octave=12, image_data_format='default', **kwargs)
```

#### Notes
    Input/output are 2D image format.
    E.g., if channel_first,
        - input_shape: ``(None, n_ch, n_freqs, n_time)``
        - output_shape: ``(None, n_ch, n_mels, n_time)``
#### Parameters
    * n_fbs: int
       - Number of filterbanks
    * sr: int
        - sampling rate. It is used to initialize `freq_to_mel`.
    * init: str
        - if `'mel'`, init with mel center frequencies and stds.
    * fmin: float
        - min frequency of filterbanks.
        - If `init == 'log'`, fmin should be > 0. Use `None` if you got no idea.
    * fmax: float
        - max frequency of filterbanks.
        - If `init == 'log'`, fmax is ignored.
    * trainable_fb: bool,
        - Whether the filterbanks are trainable or not.

### `AdditiveNoise`
([↑up to contents](#contents))

```python
kapre.augmentation.AdditiveNoise(power=0.1, random_gain=False, noise_type='white', **kwargs)
```

Add noise to input data and output it.

#### Parameters

    * power: float [scalar]
        - The power of noise. std if it's white noise.
        - Default: `0.1`
    * random_gain: bool
        - Whether the noise gain is random or not.
        - If `True`, gain is sampled from `uniform(low=0.0, high=power)` in every batch.
        - Default: `False`
    * noise_type; str,
        - Specify the type of noise. It only supports `'white'` now.
        - Default: `white`


#### Returns

Same shape as input data but with additional generated noise.
