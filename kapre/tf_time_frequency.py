import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from . import backend, backend_keras
import tensorflow as tf 
import warnings

class Spectrogram(Layer):
    """
    ### `Spectrogram`

    ```python
    kapre.time_frequency.Spectrogram(n_dft=512, n_hop=None, padding='same',
                                     power_spectrogram=2.0, return_decibel_spectrogram=False,
                                     trainable_kernel=False, 
                                     image_data_format='default',
                                     **kwargs)
    ```
    Spectrogram layer that outputs spectrogram(s) in 2D image format.

    #### Parameters
     * n_fft: int > 0 [scalar]
       - The number of fFT points, presumably power of 2.
       - Default: ``512``

     * hop_length: int > 0 [scalar]
       - Hop length between frames in sample,  probably <= ``n_fft``.
       - Default: ``None`` (``n_fft //4 `` is used)

     * pad_mode: str, 'default', 'channels_first', 'channels_last'.
       - Padding strategies at the ends of signal.
       - Default: ``'constant'``
       
     * center: bool, 
       - If center== True pad_mode is used as an argument in tf.pad 
       - If center ==False pad_mode is ignored
       - Default '''True'''
       
     * power_spectrogram: float [scalar],
       -  ``2.0`` to get power-spectrogram, ``1.0`` to get power-spectrogram.
       -  Usually ``1.0`` or ``2.0``.
       -  Default: ``2.0``

     * return_decibel_spectrogram: bool,
       -  Whether to return in decibel or not, i.e. returns log10(amplitude spectrogram) if ``True``.
       -  Recommended to use ``True``, although it's not by default.
       -  Default: ``False``

     * keep_old_order: bool
       -  Whether return (None, n_channel, n_freq, n_time) instead of 
       (None, n_channel, , n_time, n_freq)[newer version] if `'channels_first'`
       - Whether return (None, n_freq, n_time,n_channel) instead of 
       (None, , n_time, n_freq, n_channel)[newer version] if `'channels_last'``

     * image_data_format: string, ``'channels_first'`` or ``'channels_last'``.
       -  The returned spectrogram follows this image_data_format strategy.
       -  If ``'default'``, it follows the current Keras session's setting.
       -  Setting is in ``./keras/keras.json``.
       -  Default: ``'default'``

    #### Notes
     * The input should be a 2D array, ``(audio_channel, audio_length)``.
     * E.g., ``(1, 44100)`` for mono signal, ``(2, 44100)`` for stereo signal.
     * It supports multichannel signal input, so ``audio_channel`` can be any positive integer.
     * The input shape is not related to keras `image_data_format()` config.

    #### Returns

    A Keras layer

     * abs(Spectrogram) in a shape of 2D data, i.e.,
     * output order is up to kee_old_order flag
     * `(None, n_channel, n_freq, n_time)` if `'channels_first'`,
     * `(None, n_freq, n_time, n_channel)` if `'channels_last'`,


    """

    def __init__(
        self,
        n_fft:int=512,
        hop_length:int=None,
        center:bool=True,
        pad_mode:str='constant',
        power_spectrogram:float=1.0,
        return_decibel_spectrogram:bool=False,
        keep_old_order:bool=False,
        image_data_format:str='default',
        **kwargs,
    ):
        assert n_fft > 1 and ((n_fft & (n_fft - 1)) == 0), (
            'n_fft should be > 1 and power of 2, but n_fft == %d' % n_fft
        )
        assert pad_mode in ('symmetric', 'reflect', 'constant')
        assert n_fft % 2 == 0
        assert image_data_format in ('default', 'channels_first', 'channels_last')
        
            
        if hop_length is None:
            hop_length = n_fft // 4
        
        if image_data_format == 'default':
            self.image_data_format = K.image_data_format()
        else:
            self.image_data_format = image_data_format
        
        self.keep_old_order = keep_old_order
        
        if self.keep_old_order:
            tf.python.deprecation.logging.warn("(..,Features,Time,..) output order is deprecated (..,Time,Features,..) is the future \\o/")
            
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.power_spectrogram = power_spectrogram
        self.return_decibel_spectrogram = return_decibel_spectrogram
        
        
        super(Spectrogram, self).__init__(**kwargs)

    def build(self, input_shape):
        
        self.n_ch = input_shape[1]
        self.len_src = input_shape[2]
        self.is_mono = self.n_ch == 1
        
        if self.image_data_format == 'channels_first':
            self.ch_axis_idx = 1
        else:
            self.ch_axis_idx = 3
        
        if self.len_src is not None:
            assert self.len_src >= self.n_fft, 'Input is too short!'
        

        super(Spectrogram, self).build(input_shape)

    def call(self, x):
        output = self._tf_get_stft(x, 
                                   n_fft=self.n_fft, 
                                   hop_length=self.hop_length,
                                   center=self.center,
                                   pad_mode=self.pad_mode)
        
        output = K.pow(K.abs(output), self.power_spectrogram)
        
        if self.return_decibel_spectrogram:
            output = backend_keras.amplitude_to_decibel(output)
        
        if self.keep_old_order:
            output = self._gone_in_future_version_order(output)
            
        if self.image_data_format == 'channels_last':
            output = K.permute_dimensions(output, [0, 2, 3, 1])
        
        
            
        
        return output
    
    def _gone_in_future_version_order(self, output):
        return K.permute_dimensions(output, [0, 1, 3, 2])

    def get_config(self):
        config = {
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'pad_mode': self.pad_mode,
            'center': self.center,
            'power_spectrogram': self.power_spectrogram,
            'return_decibel_spectrogram': self.return_decibel_spectrogram,
            'keep_old_order': self.keep_old_order,
            'image_data_format': self.image_data_format,
        }
        base_config = super(Spectrogram, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def _tf_get_stft(self, x, 
                     n_fft:int, 
                     hop_length:int, 
                     center:bool=True,
                     pad_mode:str='reflect'):
        
        assert pad_mode in ('symmetric', 'reflect', 'constant')
        
        if center:
            x = tf.pad(x, tf.constant([[0,0],[0,0],[n_fft//2,n_fft//2]]), mode=pad_mode)

        return tf.signal.stft(
                x,
                frame_length=n_fft,
                frame_step=hop_length,
                fft_length=n_fft,
                pad_end=False,
                window_fn=tf.signal.hann_window)



class MelSpectrogram(Spectrogram):
    """
    ### `Melspectrogram`
    ```python
    kapre.time_frequency.Melspectrogram(sr=22050, n_mels=128, fmin=0.0, fmax=None,
                                        power_melgram=1.0, return_decibel_melgram=False,
                                        trainable_fb=False, **kwargs)
    ```

    Mel-spectrogram layer that outputs mel-spectrogram(s) in 2D image format.

    Its base class is ``Spectrogram``.

    Mel-spectrogram is an efficient representation using the property of human
    auditory system -- by compressing frequency axis into mel-scale axis.

    #### Parameters
     * sr: integer > 0 [scalar]
       - sampling rate of the input audio signal.
       - Default: ``16000``

     * n_mels: int > 0 [scalar]
       - The number of mel bands.
       - Default: ``128``

     * fmin: float > 0 [scalar]
       - Minimum frequency to include in Mel-spectrogram.
       - Default: ``0.0``

     * fmax: float > ``fmin`` [scalar]
       - Maximum frequency to include in Mel-spectrogram.
       - If `None`, it is inferred as ``sr / 2``.
       - Default: `None`

     * power_melgram: float [scalar]
       - Power of ``2.0`` if power-spectrogram,
       - ``1.0`` if amplitude spectrogram.
       - Default: ``1.0``

     * return_decibel_melgram: bool
       - Whether to return in decibel or not, i.e. returns log10(amplitude spectrogram) if ``True``.
       - Recommended to use ``True``, although it's not by default.
       - Default: ``False``

     * trainable_fb: bool
       - Whether the spectrogram -> mel-spectrogram filterbanks are trainable.
       - If ``True``, the frequency-to-mel matrix is initialised with mel frequencies but trainable.
       - If ``False``, it is initialised and then frozen.
       - Default: `False`

     * htk: by default in Tensorflow
     
     * **kwargs:
       - The keyword arguments of ``Spectrogram`` such as ``n_fft``, ``hop_length``,
       - ``pad_mode``, ``image_data_format``.

    #### Notes
     * The input should be a 2D array, ``(audio_channel, audio_length)``.
    E.g., ``(1, 44100)`` for mono signal, ``(2, 44100)`` for stereo signal.
     * It supports multichannel signal input, so ``audio_channel`` can be any positive integer.
     * The input shape is not related to keras `image_data_format()` config.

    #### Returns

    A Keras layer
     * abs(mel-spectrogram) in a shape of 2D data, i.e.,
     * `(None, n_channel, n_time, n_mels)` if `'channels_first'`,
     * `(None, n_time, n_mels, n_channel)` if `'channels_last'`,

    """

    def __init__(
        self,
        sr:int=16000,
        n_mels:int=128,
        fmin:float=0.0,
        fmax:float=None,
        return_decibel_melgram:bool=False,
        trainable_fb:bool=False,
        keep_old_order:bool=False,
        **kwargs,
    ):
        if 'hop_length' not in kwargs:
            print('hop_length not declared explicitly, setting it up as sr//100')
            kwargs['hop_length'] = sr // 100 # 10ms window
        
        if keep_old_order:
            tf.python.deprecation.logging.warn("(..,Features,Time,..) output order is deprecated..(..,Time,Features,..) is the future \\o/")
            
        assert sr > 0
        
        if 'power_spectrogram' in kwargs:
            assert (
                kwargs['power_spectrogram'] == 2.0
            ), 'In Melspectrogram, power_spectrogram should be set as 2.0.'
        
        if 'return_decibel_spectrogram' in kwargs:
            assert (
                kwargs['retun_decibel_spectrogram'] == False
            ), 'In Melspectrogram, return_decibel_spectrogram shoul be set False'
        
        self.fmax = sr/2.0 if fmax is None else fmax     
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.return_decibel_melgram = return_decibel_melgram
        self.trainable_fb = trainable_fb
        self.keep_old_order_mel = keep_old_order
        
        super(MelSpectrogram, self).__init__(keep_old_order=False,
                                             **kwargs)
        print(self.get_config())

    def build(self, input_shape):
        super(MelSpectrogram, self).build(input_shape)
        
        self.built = False
        # compute freq2mel matrix -->
        mel_basis = backend.tf_mel(sr=self.sr, 
                                n_fft=self.n_fft, 
                                n_mels=self.n_mels, 
                                fmin=self.fmin, 
                                fmax=self.fmax 
                                )
        
        self.freq2mel = tf.Variable(mel_basis, 
                                    dtype=K.floatx(), 
                                    trainable=self.trainable_fb)
        self.built = True

    def call(self, x):
        power_spectrogram = super(MelSpectrogram, self).call(x)
        # now,  channels_first: (batch_sample, n_ch, n_time, n_freq)
        #       channels_last: (batch_sample, n_time, n_freq, n_ch)
        
        if self.image_data_format == 'channels_last':
            power_spectrogram = K.permute_dimensions(power_spectrogram, [0, 3, 1, 2])
            # now, whatever image_data_format, (batch_sample, n_ch, n_time, n_freq)
            
        output = K.dot(power_spectrogram, self.freq2mel)
        
        if self.return_decibel_melgram:
            output = backend_keras.amplitude_to_decibel(output)

        if self.keep_old_order_mel:
            output = self._gone_in_future_version_order(output)
        
        if self.image_data_format == 'channels_last':
            output = K.permute_dimensions(output, [0, 2, 3, 1])

            
        return output

    def get_config(self):
        config = {
            'sr': self.sr,
            'n_mels': self.n_mels,
            'fmin': self.fmin,
            'fmax': self.fmax,
            'trainable_fb': self.trainable_fb,
            'return_decibel_melgram': self.return_decibel_melgram,
            'keep_old_order_mel': self.keep_old_order_mel,
        }
        base_config = super(MelSpectrogram, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
    """Determines output length of a convolution given input length.
    # Arguments
        input_length: integer.
        filter_size: integer.
        padding: one of `"same"`, `"valid"`, `"full"`.
        stride: integer.
        dilation: dilation rate, integer.
    # Returns
        The output length (integer).
    """
    if input_length is None:
        return None
    assert padding in {'same', 'valid', 'full', 'causal'}
    dilated_filter_size = (filter_size - 1) * dilation + 1
    if padding == 'same':
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'causal':
        output_length = input_length
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride

