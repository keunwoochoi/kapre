Quick Start
===========

Layers
------

Audio preprocessing layers
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Basic layers in [time_frequency.py](https://github.com/keunwoochoi/kapre/blob/master/kapre/time_frequency.py)
  #. `STFT`
  #.  `Magnitude`
  #.  `Phase`
  #.  `MagnitudeToDecibel`
  #.  `ApplyFilterbank`
  #.  `Delta` 
* Complicated layers are composed using time-frequency layers as in [composed.py](https://github.com/keunwoochoi/kapre/blob/master/kapre/composed.py).
  #.  `kapre.composed.get_perfectly_reconstructing_stft_istft()`
  #.  `kapre.composed.get_stft_mag_phase()`
  #.  `kapre.composed.get_melspectrogram_layer()`
  #.  `kapre.composed.get_log_frequency_spectrogram_layer()`. 
  
One-shot example
^^^^^^^^^^^^^^^^

.. code-block:: python

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

* See the Jupyter notebook at the [example folder](https://github.com/keunwoochoi/kapre/tree/master/examples)
