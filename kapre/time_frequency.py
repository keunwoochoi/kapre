# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
import keras
from keras import backend as K
from keras.engine import Layer
from keras.utils.conv_utils import conv_output_length
from . import backend, backend_keras


class Spectrogram(Layer):
    """
    ### `Spectrogram`

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
       - Default: ``512``

     * n_hop: int > 0 [scalar]
       - Hop length between frames in sample,  probably <= ``n_dft``.
       - Default: ``None`` (``n_dft / 2`` is used)

     * padding: str, ``'same'`` or ``'valid'``.
       - Padding strategies at the ends of signal.
       - Default: ``'same'``

     * power_spectrogram: float [scalar],
       -  ``2.0`` to get power-spectrogram, ``1.0`` to get amplitude-spectrogram.
       -  Usually ``1.0`` or ``2.0``.
       -  Default: ``2.0``

     * return_decibel_spectrogram: bool,
       -  Whether to return in decibel or not, i.e. returns log10(amplitude spectrogram) if ``True``.
       -  Recommended to use ``True``, although it's not by default.
       -  Default: ``False``

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
     * The input shape is not related to keras `image_data_format()` config.

    #### Returns

    A Keras layer

     * abs(Spectrogram) in a shape of 2D data, i.e.,
     * `(None, n_channel, n_freq, n_time)` if `'channels_first'`,
     * `(None, n_freq, n_time, n_channel)` if `'channels_last'`,


    """

    def __init__(self, n_dft=512, n_hop=None, padding='same',
                 power_spectrogram=2.0, return_decibel_spectrogram=False,
                 trainable_kernel=False, image_data_format='default', **kwargs):
        assert n_dft > 1 and ((n_dft & (n_dft - 1)) == 0), \
            ('n_dft should be > 1 and power of 2, but n_dft == %d' % n_dft)
        assert isinstance(trainable_kernel, bool)
        assert isinstance(return_decibel_spectrogram, bool)
        assert padding in ('same', 'valid')
        if n_hop is None:
            n_hop = n_dft // 2

        assert image_data_format in ('default', 'channels_first', 'channels_last')
        if image_data_format == 'default':
            self.image_data_format = K.image_data_format()
        else:
            self.image_data_format = image_data_format

        self.n_dft = n_dft
        assert n_dft % 2 == 0
        self.n_filter = n_dft // 2 + 1
        self.trainable_kernel = trainable_kernel
        self.n_hop = n_hop
        self.padding = padding
        self.power_spectrogram = float(power_spectrogram)
        self.return_decibel_spectrogram = return_decibel_spectrogram
        super(Spectrogram, self).__init__(**kwargs)

    def build(self, input_shape):
        self.n_ch = input_shape[1]
        self.len_src = input_shape[2]
        self.is_mono = (self.n_ch == 1)
        if self.image_data_format == 'channels_first':
            self.ch_axis_idx = 1
        else:
            self.ch_axis_idx = 3
        if self.len_src is not None:
            assert self.len_src >= self.n_dft, 'Hey! The input is too short!'

        self.n_frame = conv_output_length(self.len_src,
                                          self.n_dft,
                                          self.padding,
                                          self.n_hop)

        dft_real_kernels, dft_imag_kernels = backend.get_stft_kernels(self.n_dft)
        self.dft_real_kernels = K.variable(dft_real_kernels, dtype=K.floatx(), name="real_kernels")
        self.dft_imag_kernels = K.variable(dft_imag_kernels, dtype=K.floatx(), name="imag_kernels")
        # kernels shapes: (filter_length, 1, input_dim, nb_filter)?
        if self.trainable_kernel:
            self.trainable_weights.append(self.dft_real_kernels)
            self.trainable_weights.append(self.dft_imag_kernels)
        else:
            self.non_trainable_weights.append(self.dft_real_kernels)
            self.non_trainable_weights.append(self.dft_imag_kernels)

        super(Spectrogram, self).build(input_shape)
        # self.built = True

    def compute_output_shape(self, input_shape):
        if self.image_data_format == 'channels_first':
            return input_shape[0], self.n_ch, self.n_filter, self.n_frame
        else:
            return input_shape[0], self.n_filter, self.n_frame, self.n_ch

    def call(self, x):
        output = self._spectrogram_mono(x[:, 0:1, :])
        if self.is_mono is False:
            for ch_idx in range(1, self.n_ch):
                output = K.concatenate((output,
                                        self._spectrogram_mono(x[:, ch_idx:ch_idx + 1, :])),
                                       axis=self.ch_axis_idx)
        if self.power_spectrogram != 2.0:
            output = K.pow(K.sqrt(output), self.power_spectrogram)
        if self.return_decibel_spectrogram:
            output = backend_keras.amplitude_to_decibel(output)
        return output

    def get_config(self):
        config = {'n_dft': self.n_dft,
                  'n_hop': self.n_hop,
                  'padding': self.padding,
                  'power_spectrogram': self.power_spectrogram,
                  'return_decibel_spectrogram': self.return_decibel_spectrogram,
                  'trainable_kernel': self.trainable_kernel,
                  'image_data_format': self.image_data_format}
        base_config = super(Spectrogram, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _spectrogram_mono(self, x):
        '''x.shape : (None, 1, len_src),
        returns 2D batch of a mono power-spectrogram'''
        x = K.permute_dimensions(x, [0, 2, 1])
        x = K.expand_dims(x, 3)  # add a dummy dimension (channel axis)
        subsample = (self.n_hop, 1)
        output_real = K.conv2d(x, self.dft_real_kernels,
                               strides=subsample,
                               padding=self.padding,
                               data_format='channels_last')
        output_imag = K.conv2d(x, self.dft_imag_kernels,
                               strides=subsample,
                               padding=self.padding,
                               data_format='channels_last')
        output = output_real ** 2 + output_imag ** 2
        # now shape is (batch_sample, n_frame, 1, freq)
        if self.image_data_format == 'channels_last':
            output = K.permute_dimensions(output, [0, 3, 1, 2])
        else:
            output = K.permute_dimensions(output, [0, 2, 3, 1])
        return output


class Melspectrogram(Spectrogram):
    '''
    ### `Melspectrogram`
    ```python
    kapre.time_frequency.Melspectrogram(sr=22050, n_mels=128, fmin=0.0, fmax=None,
                                        power_melgram=1.0, return_decibel_melgram=False,
                                        trainable_fb=False, **kwargs)
    ```
d
    Mel-spectrogram layer that outputs mel-spectrogram(s) in 2D image format.

    Its base class is ``Spectrogram``.

    Mel-spectrogram is an efficient representation using the property of human
    auditory system -- by compressing frequency axis into mel-scale axis.

    #### Parameters
     * sr: integer > 0 [scalar]
       - sampling rate of the input audio signal.
       - Default: ``22050``

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

     * htk: bool
       - Check out Librosa's `mel-spectrogram` or `mel` option.

     * norm: float [scalar]
       - Check out Librosa's `mel-spectrogram` or `mel` option.

     * **kwargs:
       - The keyword arguments of ``Spectrogram`` such as ``n_dft``, ``n_hop``,
       - ``padding``, ``trainable_kernel``, ``image_data_format``.

    #### Notes
     * The input should be a 2D array, ``(audio_channel, audio_length)``.
    E.g., ``(1, 44100)`` for mono signal, ``(2, 44100)`` for stereo signal.
     * It supports multichannel signal input, so ``audio_channel`` can be any positive integer.
     * The input shape is not related to keras `image_data_format()` config.

    #### Returns

    A Keras layer
     * abs(mel-spectrogram) in a shape of 2D data, i.e.,
     * `(None, n_channel, n_mels, n_time)` if `'channels_first'`,
     * `(None, n_mels, n_time, n_channel)` if `'channels_last'`,

    '''

    def __init__(self,
                 sr=22050, n_mels=128, fmin=0.0, fmax=None,
                 power_melgram=1.0, return_decibel_melgram=False,
                 trainable_fb=False, htk=False, norm=1, **kwargs):

        super(Melspectrogram, self).__init__(**kwargs)
        assert sr > 0
        assert fmin >= 0.0
        if fmax is None:
            fmax = float(sr) / 2
        assert fmax > fmin
        assert isinstance(return_decibel_melgram, bool)
        if 'power_spectrogram' in kwargs:
            assert kwargs['power_spectrogram'] == 2.0, \
                'In Melspectrogram, power_spectrogram should be set as 2.0.'

        self.sr = int(sr)
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.return_decibel_melgram = return_decibel_melgram
        self.trainable_fb = trainable_fb
        self.power_melgram = power_melgram
        self.htk = htk
        self.norm = norm

    def build(self, input_shape):
        super(Melspectrogram, self).build(input_shape)
        self.built = False
        # compute freq2mel matrix --> 
        mel_basis = backend.mel(self.sr, self.n_dft, self.n_mels, self.fmin, self.fmax,
                                self.htk, self.norm)  # (128, 1025) (mel_bin, n_freq)
        mel_basis = np.transpose(mel_basis)

        self.freq2mel = K.variable(mel_basis, dtype=K.floatx())
        if self.trainable_fb:
            self.trainable_weights.append(self.freq2mel)
        else:
            self.non_trainable_weights.append(self.freq2mel)
        self.built = True

    def compute_output_shape(self, input_shape):
        if self.image_data_format == 'channels_first':
            return input_shape[0], self.n_ch, self.n_mels, self.n_frame
        else:
            return input_shape[0], self.n_mels, self.n_frame, self.n_ch

    def call(self, x):
        power_spectrogram = super(Melspectrogram, self).call(x)
        # now,  channels_first: (batch_sample, n_ch, n_freq, n_time)
        #       channels_last: (batch_sample, n_freq, n_time, n_ch)
        if self.image_data_format == 'channels_first':
            power_spectrogram = K.permute_dimensions(power_spectrogram, [0, 1, 3, 2])
        else:
            power_spectrogram = K.permute_dimensions(power_spectrogram, [0, 3, 2, 1])
        # now, whatever image_data_format, (batch_sample, n_ch, n_time, n_freq)
        output = K.dot(power_spectrogram, self.freq2mel)
        if self.image_data_format == 'channels_first':
            output = K.permute_dimensions(output, [0, 1, 3, 2])
        else:
            output = K.permute_dimensions(output, [0, 3, 2, 1])
        if self.power_melgram != 2.0:
            output = K.pow(K.sqrt(output), self.power_melgram)
        if self.return_decibel_melgram:
            output = backend_keras.amplitude_to_decibel(output)
        return output

    def get_config(self):
        config = {'sr': self.sr,
                  'n_mels': self.n_mels,
                  'fmin': self.fmin,
                  'fmax': self.fmax,
                  'trainable_fb': self.trainable_fb,
                  'power_melgram': self.power_melgram,
                  'return_decibel_melgram': self.return_decibel_melgram,
                  'htk': self.htk,
                  'norm': self.norm}
        base_config = super(Melspectrogram, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
