.. Kapre documentation master file, created by
   sphinx-quickstart on Thu Aug 27 19:40:10 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Kapre's documentation!
=================================

Keras Audio Preprocessors - compute STFT, ISTFT, Melspectrogram, and others on GPU real-time.
  
Tested on Python 3.3, 3.6, and 3.7.

Why?
----
* Kapre enables you to optimize DSP parameters and makes model deployment simpler with less dependency.  
* Kapre layers are consistent with 1D/2D tensorflow batch shapes.
* Kapre layers are compatible with `'channels_first'` and `'channels_last'`
* Kapre layers are tested against Librosa (stft, decibel, etc) - which is (trust me) *tricker* than you think.
* Kapre layers have extended APIs from the default `tf.signals` implementation.
* Kapre provides a perfectly invertible `STFT` and `InverseSTFT` pair.
* You save your time implementing and testing all of these.
* Kapre is available on pip with versioning; hence you keep your code reproducible.   

Installation
------------
 
.. code-block:: none

  pip install kapre


Example
-------

See the Jupyter notebook at the [example folder](https://github.com/keunwoochoi/kapre/tree/master/examples)

Citation
--------

Please cite this paper if you use Kapre for your work.

.. code-block:: none

  @inproceedings{choi2017kapre,
    title={Kapre: On-GPU Audio Preprocessing Layers for a Quick Implementation of Deep Neural Network Models with Keras},
    author={Choi, Keunwoo and Joo, Deokjin and Kim, Juho},
    booktitle={Machine Learning for Music Discovery Workshop at 34th International Conference on Machine Learning},
    year={2017},
    organization={ICML}
  }

Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
