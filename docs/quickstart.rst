Examples
========

We provide fully functioning code snippets here.
More detailed examples are under documentations of all the layers and functions.


How To Import
-------------

.. code-block:: python

    import kapre  # to import the whole library
    from kapre import (  # `time_frequency` layers can be directly imported from `kapre`
        STFT,
        InverseSTFT,
        Magnitude,
        Phase,
        MagnitudeToDecibel,
        ApplyFilterbank,
        Delta,
        ConcatenateFrequencyMap,
    )
    from kapre import (  # `signal` layers can be also directly imported from kapre
        Frame,
        Energy,
        MuLawEncoding,
        MuLawDecoding,
        LogmelToMFCC,
    )
    # from kapre import backend  # we can do this, but `backend` might be a too general name
    import kapre.backend  # for namespace sanity, you might prefer this
    from kapre import backend as kapre_backend  # or maybe this
    from kapre.composed import (  # function names in `composed` are purposefully verbose.
        get_stft_magnitude_layer,
        get_melspectrogram_layer,
        get_log_frequency_spectrogram_layer,
        get_perfectly_reconstructing_stft_istft,
        get_stft_mag_phase,
        get_frequency_aware_conv2d,
    )

Use STFT Magnitude
------------------

.. code-block:: python

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from kapre import STFT, Magnitude, MagnitudeToDecibel

    sampling_rate = 16000  # sampling rate of your input audio
    duration = 20.0  # duration of the audio
    num_channel = 2  # number of channels of the audio
    input_shape = (int(sampling_rate * duration), num_channel)  # let's follow `channels_last` convention

    model = Sequential()
    model.add(STFT(n_fft=2048, win_length=2018, hop_length=1024,
                   window_name='hann_window', pad_end=False,
                   input_data_format='channels_last', output_data_format='channels_last',
                   input_shape=input_shape))  # complex64
    model.add(Magnitude())   # float32
    model.add(MagnitudeToDecibel())  # float32 but in decibel scale
    model.summary()  # this would be an "audio frontend" of your model
    """
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    stft (STFT)                  (None, 311, 1025, 2)      0
    _________________________________________________________________
    magnitude (Magnitude)        (None, 311, 1025, 2)      0
    _________________________________________________________________
    magnitude_to_decibel (Magnit (None, 311, 1025, 2)      0
    =================================================================
    Total params: 0
    Trainable params: 0
    Non-trainable params: 0
    _________________________________________________________________
    """
    # A 20-second stereo audio signal is converted to a (311, 1025, 2) tensor.

    # Now, you can add your own model. For example, let's add ResNet50
    # with global average pooling, no pre-trained weights,
    # and for a 10-class classification.

    model.add(
        tf.keras.applications.ResNet50(
            include_top=True, weights=None, input_tensor=None,
            input_shape=(311, 1025, 2), pooling='avg', classes=10
        )
    )

    model.summary()
    """
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    stft (STFT)                  (None, 311, 1025, 2)      0
    _________________________________________________________________
    magnitude (Magnitude)        (None, 311, 1025, 2)      0
    _________________________________________________________________
    magnitude_to_decibel (Magnit (None, 311, 1025, 2)      0
    _________________________________________________________________
    resnet50 (Functional)        (None, 10)                23605066
    =================================================================
    Total params: 23,605,066
    Trainable params: 23,551,946
    Non-trainable params: 53,120
    _________________________________________________________________
    """



Use STFT Magnitude -- a lazy version
------------------------------------

.. code-block:: python

    from tensorflow.keras.models import Sequential
    from kapre.composed import get_stft_magnitude_layer

    sampling_rate = 16000  # sampling rate of your input audio
    duration = 20.0  # duration of the audio
    num_channel = 2  # number of channels of the audio
    input_shape = (int(sampling_rate * duration), num_channel)  # let's follow `channels_last` convention

    model = Sequential(get_stft_magnitude_layer(input_shape=input_shape, return_decibel=True))

    model.summary()  # this lazy version provides an abstraction view of stft_magnitude
    """
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    stft_magnitude (Sequential)  (None, 622, 1025, 2)      0
    =================================================================
    Total params: 0
    Trainable params: 0
    Non-trainable params: 0
    _________________________________________________________________
    """
    # Here, a 20-second stereo audio signal is converted to a (622, 1025, 2) tensor.
    # x2 more temporal frames compared to the example above because we didn't set hop_length here,
    # and that means it's set to a 25% hop length, not 50% as above.

    model.layers[0].summary()  # let's deep dive one level
    """
    Model: "stft_magnitude"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    stft (STFT)                  (None, 622, 1025, 2)      0
    _________________________________________________________________
    magnitude (Magnitude)        (None, 622, 1025, 2)      0
    _________________________________________________________________
    magnitude_to_decibel (Magnit (None, 622, 1025, 2)      0
    =================================================================
    Total params: 0
    Trainable params: 0
    Non-trainable params: 0
    _________________________________________________________________
    """



