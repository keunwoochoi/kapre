Release Note
^^^^^^^^^^^^

* 18 March 2021
  - 0.3.5
    - Add `kapre.time_frequency_tflite` which uses tflite for a faster CPU inference.

* 29 Sep 2020
  - 0.3.4
    - Fix a bug in `kapre.backend.get_window_fn()`. Previously, it only correctly worked with `None` input and an error was raised when non-default value was set for `window_name` in any layer.

* 15 Sep 2020
  - 0.3.3
    - `kapre.augmentation` is added
    - `kapre.time_frequency.ConcatenateFrequencyMap` is added
    - `kapre.composed.get_frequency_aware_conv2d` is added
    - In `STFT` and `InverseSTFT`, keyword arg `window_fn` is renamed to `window_name` and it expects string value, not function.
      - With this update, models with Kapre layers can be loaded with `h5` file format.
    - `kapre.backend.get_window_fn` is added

* 28 Aug 2020
  - 0.3.2
    - `kapre.signal.Frame` and `kapre.signal.Energy` are added
    - `kapre.signal.LogmelToMFCC` is added
    - `kapre.signal.MuLawEncoder` and `kapre.signal.MuLawDecoder` are added
    - `kapre.composed.get_stft_magnitude_layer()` is added
* 21 Aug 2020
  - 0.3.1
    - `Inverse STFT` is added

* 15 Aug 2020
  - 0.3.0
    - Breaking and simplifying changes with Tensorflow 2.0 and more tests. Some features are removed.

* 29 Jul 2020
  - 0.2.0
    - Change melspectrogram filterbank from `norm=1` to `norm='slaney'` (w.r.t. Librosa) due to the update from Librosa (https://github.com/keunwoochoi/kapre/issues/77)
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

* 20 Feb 2019
  - Kapre ver 0.1.4
    - Fixed amplitude-to-decibel error as raised in https://github.com/keunwoochoi/kapre/issues/46

* March 2018
  - Kapre ver 0.1.3
    - Kapre is on Pip again
    - Add unit tests
    - Remove `Datasets`
    - Remove some codes while adding more dependency on Librosa to make it cleaner and more stable
      - and therefore `htk` option enabled in `Melspectrogram`

* 9 July 2017
  - Kapre ver 0.1.1, aka 'pretty stable' with a benchmark paper, https://arxiv.org/abs/1706.05781
    - Remove STFT, python3 compatible
    - A full documentation in this readme.md
    - pip version is updated