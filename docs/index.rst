.. Kapre documentation master file, created by
   sphinx-quickstart on Thu Aug 27 19:40:10 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Kapre
=====

Keras Audio Preprocessors - compute STFT, InverseSTFT, Melspectrogram, and others on GPU real-time.
  
Tested on Python 3.6, and 3.7.

Why Kapre?
----------

vs. Pre-computation
^^^^^^^^^^^^^^^^^^^

* You can optimize DSP parameters
* Your model deployment becomes much simpler and consistent.
* Your code and model has less dependencies

vs. Your own implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Quick and easy!
* Consistent with 1D/2D tensorflow batch shapes
* Data format agnostic (`channels_first` and `channels_last`)
* Less error prone - Kapre layers are tested against Librosa (stft, decibel, etc) - which is (trust me) *trickier* than you think.
* Kapre layers have some extended APIs from the default `tf.signals` implementation such as..
  - A perfectly invertible `STFT` and `InverseSTFT` pair
  - Mel-spectrogram with more options
* Reproducibility - Kapre is available on pip with versioning

Workflow with Kapre
-------------------

1. Preprocess your audio dataset. Resample the audio to the right sampling rate and store the audio signals (waveforms).
2. In your ML model, add Kapre layer e.g. `kapre.time_frequency.STFT()` as the first layer of the model.
3. The data loader simply loads audio signals and feed them into the model
4. In your hyperparameter search, include DSP parameters like `n_fft` to boost the performance.
5. When deploying the final model, all you need to remember is the sampling rate of the signal. No dependency or preprocessing!

Installation
------------
 
.. code-block:: none

  pip install kapre


Example
-------

See the Jupyter notebook at the `example folder <https://github.com/keunwoochoi/kapre/tree/master/examples>`_


Citation
--------

Please cite `this paper <https://arxiv.org/abs/1706.05781>`_ if you use Kapre for your work.


.. code-block:: none

  @inproceedings{choi2017kapre,
    title={Kapre: On-GPU Audio Preprocessing Layers for a Quick Implementation of Deep Neural Network Models with Keras},
    author={Choi, Keunwoo and Joo, Deokjin and Kim, Juho},
    booktitle={Machine Learning for Music Discovery Workshop at 34th International Conference on Machine Learning},
    year={2017},
    organization={ICML}
  }


Contribution
------------
Visit `github.com/keunwoochoi/kapre <https://github.com/keunwoochoi/kapre>`_ and chat with us :)


.. toctree::
   :hidden:
   :caption: Kapre

   quickstart


.. toctree::
   :hidden:
   :caption: API

   time_frequency
   signal
   composed
   backend
   time_frequency_tflite

.. toctree::
   :hidden:
   :caption: Info

   release_note

