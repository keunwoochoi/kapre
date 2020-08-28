# Kapre
Keras Audio Preprocessors - compute STFT, ISTFT, Melspectrogram, and others on GPU real-time.
  
Tested on Python 3.6 and 3.7

## Why?
- Kapre enables you to optimize DSP parameters and makes model deployment simpler with less dependency.  
- Kapre layers are consistent with 1D/2D tensorflow batch shapes.
- Kapre layers are compatible with `'channels_first'` and `'channels_last'`
- Kapre layers are tested against Librosa (stft, decibel, etc) - which is (trust me) *tricker* than you think.
- Kapre layers have extended APIs from the default `tf.signals` implementation.
- Kapre provides a perfectly invertible `STFT` and `InverseSTFT` pair.
- You save your time implementing and testing all of these.
- Kapre is available on pip with versioning; hence you keep your code reproducible.   

## Installation
 
```sh
pip install kapre
```

## Usage
### Layers

Audio preprocessing layers
* Basic layers in [time_frequency.py](https://github.com/keunwoochoi/kapre/blob/master/kapre/time_frequency.py)
  - `STFT`
  - `Magnitude`
  - `Phase`
  - `MagnitudeToDecibel`
  - `ApplyFilterbank`
  - `Delta` 
* Complicated layers are composed using time-frequency layers as in [composed.py](https://github.com/keunwoochoi/kapre/blob/master/kapre/composed.py).
  - `kapre.composed.get_perfectly_reconstructing_stft_istft()`
  - `kapre.composed.get_stft_mag_phase()`
  - `kapre.composed.get_melspectrogram_layer()`
  - `kapre.composed.get_log_frequency_spectrogram_layer()`. 
  
(Note: Official documentation is coming soon)

## One-shot example

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Softmax
from kapre.time_frequency import STFT, Magnitude, MagnitudeToDecibel
from kapre.composed import get_melspectrogram_layer, get_log_frequency_spectrogram_layer

# 6 channels (!), maybe 1-sec audio signal, for an example.
input_shape = (6, 44100)
sr = 44100
model = Sequential()
# A STFT layer
model.add(STFT(n_fft=2048, win_length=2018, hop_length=1024,
               window_fn=None, pad_end=False,
               input_data_format='channels_last', output_data_format='channels_last',
               input_shape=input_shape))
model.add(Magnitude())
model.add(MagnitudeToDecibel())
# Alternatively, you may want to use a melspectrogram layer
# melgram_layer = get_melspectrogram_layer()
# or log-frequency layer
# log_stft_layer = get_log_frequency_spectrogram_layer() 

# add more layers as you want
model.add(Conv2D(32, (3, 3), strides=(2, 2)))
model.add(BatchNormalization())
model.add(ReLU())
model.add(GlobalAveragePooling2D())
model.add(Dense(10))
model.add(Softmax())

# Compile the model
model.compile('adam', 'categorical_crossentropy') # if single-label classification

# train it with raw audio sample inputs
# for example, you may have functions that load your data as below.
x = load_x() # e.g., x.shape = (10000, 6, 44100)
y = load_y() # e.g., y.shape = (10000, 10) if it's 10-class classification
# then..
model.fit(x, y)
# Done!
```

* See the Jupyter notebook at the [example folder](https://github.com/keunwoochoi/kapre/tree/master/examples)

# Citation

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

## News

* 15 Aug 2020
  - 0.3.0
    - Breaking and simplifying changes with Tensorflow 2.0 and more tests. Some features are removed.

* 29 Jul 2020
  - 0.2.0
    - Change melspectrogram filterbank from `norm=1` to `norm='slaney'` (w.r.t. Librosa) due to the update from Librosa ([#77](https://github.com/keunwoochoi/kapre/issues/77)). 
    This would change the behavior of melspectrogram slightly.
    - Bump librosa version to 0.7.2 or higher.
* 17 Mar 2020
  - 0.1.8
    - added `utils.Delta` layer
* 20 Feb 2020
  - Kapre ver 0.1.7
    - No vanilla Keras dependency
    - Tensorflow >= 1.15 only
    - Not tested on Python 2.7 anymore; only on Python 3.6 and 3.7 locally (by `tox`) and 3.6 on Travis 

..and more at `news.md`. 