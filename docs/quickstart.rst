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

    from tensorflow.keras.models import Sequential
    from kapre import STFT, Magnitude, MagnitudeToDecibel

    sampling_rate = 16000  # sampling rate of your input audio
    duration = 20.0  # duration of the audio
    num_channel = 2  # number of channels of the audio
    input_shape = (num_channel, int(sampling_rate * duration))  # let's follow `channels_last` convention even for audio

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
    stft (STFT)                  (None, 0, 1025, 320000)   0
    _________________________________________________________________
    magnitude (Magnitude)        (None, 0, 1025, 320000)   0
    _________________________________________________________________
    magnitude_to_decibel (Magnit (None, 0, 1025, 320000)   0
    =================================================================
    Total params: 0
    Trainable params: 0
    Non-trainable params: 0
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
    input_shape = (num_channel, int(sampling_rate * duration))  # let's follow `channels_last` convention even for audio

    model = Sequential(get_stft_magnitude_layer(input_shape=input_shape, return_decibel=True))

    model.summary()  # this lazy version provides an abstraction view of stft_magnitude
    """
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    stft_magnitude (Sequential)  (None, 0, 1025, 320000)   0
    =================================================================
    Total params: 0
    Trainable params: 0
    Non-trainable params: 0
    _________________________________________________________________
    """

    model.layers[0].summary()  # let's deep dive one level
    """
    Model: "stft_magnitude"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    stft (STFT)                  (None, 0, 1025, 320000)   0
    _________________________________________________________________
    magnitude (Magnitude)        (None, 0, 1025, 320000)   0
    _________________________________________________________________
    magnitude_to_decibel (Magnit (None, 0, 1025, 320000)   0
    =================================================================
    Total params: 0
    Trainable params: 0
    Non-trainable params: 0
    _________________________________________________________________
    """



