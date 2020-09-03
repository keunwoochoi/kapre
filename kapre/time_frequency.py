"""Time-frequency layers.

This module has implementations of some popular time-frequency operations such as STFT and inverse STFT.

"""
import warnings
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D
from . import backend
from tensorflow.keras import backend as K
from .backend import CH_FIRST_STR, CH_LAST_STR, CH_DEFAULT_STR


def _shape_spectrum_output(spectrums, data_format):
    """Shape batch spectrograms into the right format.

    Args:
        spectrums (`Tensor`): result of tf.signal.stft or similar, i.e., (..., time, freq).
        data_format (`str`): 'channels_first' or 'channels_last'

    Returns:
        spectrums (`Tensor`): a transposed version of input `spectrums`
    """
    if data_format == CH_FIRST_STR:
        pass  # probably it's already (batch, channel, time, freq)
    else:
        spectrums = tf.transpose(spectrums, perm=(0, 2, 3, 1))  # (batch, time, freq, channel)
    return spectrums


class STFT(Layer):
    """
    A Shor-time Fourier transform layer.

    It uses `tf.signal.stft` to compute complex STFT. Additionally, it reshapes the output to be a proper 2D batch.

    If `output_data_format == 'channels_last'`, the output shape is (batch, time, freq, channel)
    If `output_data_format == 'channels_first'`, the output shape is (batch, channel, time, freq)

    Args:
        n_fft (`int`): Number of FFTs. Defaults to `2048`
        win_length (`int` or `None`): Window length in sample. Defaults to `n_fft`.
        hop_length (`int` or `None`): Hop length in sample between analysis windows. Defaults to `n_fft // 4` following Librosa.
        window_fn (function or None): A function that returns a 1D tensor window that is used in analysis. Defaults to `tf.signal.hann_window`
        pad_begin(`bool`): Whether to pad with zeros along time axis (legnth: win_length - hop_length). Defaults to `False`.
        pad_end (`bool`): Whether to pad with zeros at the finishing end of the signal.
        input_data_format (`str`): the audio data format of input waveform batch.
            `'channels_last'` if it's `(batch, time, channels)`
            `'channels_first'` if it's `(batch, channels, time)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())
        output_data_format (`str`): The data format of output STFT.
            `'channels_last'` if you want `(batch, time, frequency, channels)`
            `'channels_first'` if you want `(batch, channels, time, frequency)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())

        **kwargs: Keyword args for the parent keras layer (e.g., `name`)

    """

    def __init__(
        self,
        n_fft=2048,
        win_length=None,
        hop_length=None,
        window_fn=None,
        pad_begin=False,
        pad_end=False,
        input_data_format='default',
        output_data_format='default',
        **kwargs,
    ):
        super(STFT, self).__init__(**kwargs)

        backend.validate_data_format_str(input_data_format)
        backend.validate_data_format_str(output_data_format)

        if win_length is None:
            win_length = n_fft
        if hop_length is None:
            hop_length = win_length // 4
        if window_fn is None:
            window_fn = tf.signal.hann_window

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window_fn = window_fn
        self.pad_begin = pad_begin
        self.pad_end = pad_end

        idt, odt = input_data_format, output_data_format
        self.output_data_format = K.image_data_format() if odt == CH_DEFAULT_STR else odt
        self.input_data_format = K.image_data_format() if idt == CH_DEFAULT_STR else idt

    def call(self, x):
        """
        Compute STFT of the input signal. If the `time` axis is not the last axis of `x`, it should be transposed first.

        Args:
            x (float `Tensor`): batch of audio signals, (batch, ch, time) or (batch, time, ch) based on input_data_format

        Return:
            (complex `Tensor`): A STFT representation of x in a 2D batch shape.
            `complex64` if `x` is `float32`. `complex128` if `x` is `float64`.
            Its shape is (batch, time, freq, ch) or (batch. ch, time, freq) depending on `output_data_format` and
                `time` is the number of frames, which is `((len_src + (win_length - hop_length) / hop_length) // win_length )` if `pad_end` is `True`.
                `freq` is the number of fft unique bins, which is `n_fft // 2 + 1` (the unique components of the FFT).
        """
        waveforms = x  # (batch, ch, time) if input_data_format == 'channels_first'.
        # (batch, time, ch) if input_data_format == 'channels_last'.

        # this is needed because tf.signal.stft lives in channels_first land.
        if self.input_data_format == CH_LAST_STR:
            waveforms = tf.transpose(
                waveforms, perm=(0, 2, 1)
            )  # always (batch, ch, time) from here

        if self.pad_begin:
            waveforms = tf.pad(
                waveforms, tf.constant([[0, 0], [0, 0], [int(self.n_fft - self.hop_length), 0]])
            )

        stfts = tf.signal.stft(
            signals=waveforms,
            frame_length=self.win_length,
            frame_step=self.hop_length,
            fft_length=self.n_fft,
            window_fn=self.window_fn,
            pad_end=self.pad_end,
            name='%s_tf.signal.stft' % self.name,
        )  # (batch, ch, time, freq)

        if self.output_data_format == CH_LAST_STR:
            stfts = tf.transpose(stfts, perm=(0, 2, 3, 1))  # (batch, t, f, ch)

        return stfts

    def get_config(self):
        config = super(STFT, self).get_config()
        config.update(
            {
                'n_fft': self.n_fft,
                'win_length': self.win_length,
                'hop_length': self.hop_length,
                'window_fn': self.window_fn,
                'pad_begin': self.pad_begin,
                'pad_end': self.pad_end,
                'input_data_format': self.input_data_format,
                'output_data_format': self.output_data_format,
            }
        )
        return config


class InverseSTFT(Layer):
    """An inverse-STFT layer.

    If `output_data_format == 'channels_last'`, the output shape is (batch, time, channel)
    If `output_data_format == 'channels_first'`, the output shape is (batch, channel, time)

    Note that the result of inverse STFT could be longer than the original signal due to the padding. Do check the
    size of the result by yourself and trim it if needed.

    Args:
        n_fft (`int`): Number of FFTs. Defaults to `2048`
        win_length (`int` or `None`): Window length in sample. Defaults to `n_fft`.
        hop_length (`int` or `None`): Hop length in sample between analysis windows. Defaults to `n_fft // 4` following Librosa.
        window_fn (function or `None`): A function that returns a 1D tensor window. Defaults to `tf.signal.hann_window`, but
            this default setup does NOT lead to perfect reconstruction.
            For perfect reconstruction, this should be set using `tf.signal.inverse_stft_window_fn()` with the
            correct `frame_step` and `forward_window_fn` that are matched to those used during STFT.
        input_data_format (`str`): the data format of input STFT batch
            `'channels_last'` if you want `(batch, time, frequency, channels)`
            `'channels_first'` if you want `(batch, channels, time, frequency)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())
        output_data_format (`str`): the audio data format of output waveform batch.
            `'channels_last'` if it's `(batch, time, channels)`
            `'channels_first'` if it's `(batch, channels, time)`
            Defaults to the setting of your Keras configuration. (tf.keras.backend.image_data_format())

        **kwargs: Keyword args for the parent keras layer (e.g., `name`)

    """

    def __init__(
        self,
        n_fft=2048,
        win_length=None,
        hop_length=None,
        window_fn=None,
        input_data_format='default',
        output_data_format='default',
        **kwargs,
    ):
        super(InverseSTFT, self).__init__(**kwargs)

        backend.validate_data_format_str(input_data_format)
        backend.validate_data_format_str(output_data_format)

        if win_length is None:
            win_length = n_fft
        if hop_length is None:
            hop_length = win_length // 4
        if window_fn is None:
            window_fn = tf.signal.hann_window
            warnings.warn(
                'In InverseSTFT, forward_window_fn was not set, hence the default hann window function'
                'is used. This would lead to non-perfect reconstruction of a STFT-ISTFT chain.'
                'For perfect reconstruction, forward_window_fn should be set using'
                'tf.signal.inverse_stft_window_fn with frame_step and forward_window_fn correctly specified.'
                'For more details, see how kapre.composed.get_perfectly_reconstructing_stft_istft() is'
                'implemented. '
            )

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window_fn = window_fn

        idt, odt = input_data_format, output_data_format
        self.output_data_format = K.image_data_format() if odt == CH_DEFAULT_STR else odt
        self.input_data_format = K.image_data_format() if idt == CH_DEFAULT_STR else idt

    def call(self, x):
        """
        Compute inverse STFT of the input STFT.

        Args:
            x (complex `Tensor`): batch of STFTs, (batch, ch, time, freq) or (batch, time, freq, ch) depending on `input_data_format`

        Return:
            (`float`): audio signals of x. Shape: 1D batch shape. I.e., (batch, time, ch) or (batch, ch, time) depending on `output_data_format`

        """
        stfts = x  # (batch, ch, time, freq) if input_data_format == 'channels_first'.
        # (batch, time, freq, ch) if input_data_format == 'channels_last'.

        # this is needed because tf.signal.stft lives in channels_first land.
        if self.input_data_format == CH_LAST_STR:
            stfts = tf.transpose(stfts, perm=(0, 3, 1, 2))  # now always (b, ch, t, f)

        waveforms = tf.signal.inverse_stft(
            stfts=stfts,
            frame_length=self.win_length,
            frame_step=self.hop_length,
            fft_length=self.n_fft,
            window_fn=self.window_fn,
            name='%s_tf.signal.istft' % self.name,
        )  # (batch, ch, time)

        if self.output_data_format == CH_LAST_STR:
            waveforms = tf.transpose(waveforms, perm=(0, 2, 1))  # (batch, time, ch)

        return waveforms

    def get_config(self):
        config = super(InverseSTFT, self).get_config()
        config.update(
            {
                'n_fft': self.n_fft,
                'win_length': self.win_length,
                'hop_length': self.hop_length,
                'window_fn': self.window_fn,
                'input_data_format': self.input_data_format,
                'output_data_format': self.output_data_format,
            }
        )
        return config


class Magnitude(Layer):
    """Compute the magnitude of the complex input, resulting in a float tensor"""

    def call(self, x):
        return tf.abs(x)


class Phase(Layer):
    """Compute the phase of the complex input in radian, resulting in a float tensor"""

    def call(self, x):
        return tf.math.angle(x)


class MagnitudeToDecibel(Layer):
    """A class that wraps `backend.magnitude_to_decibel` to compute decibel of the input magnitude.

    Args:
        ref_value (`float`): an input value that would become 0 dB in the result.
            For spectrogram magnitudes, ref_value=1.0 usually make the decibel-sclaed output to be around zero
            if the input audio was in [-1, 1].
        amin (`float`): the noise floor of the input. An input that is smaller than `amin`, it's converted to `amin.
        dynamic_range (`float`): range of the resulting value. E.g., if the maximum magnitude is 30 dB,
            the noise floor of the output would become (30 - dynamic_range) dB

    """

    def __init__(self, ref_value=1.0, amin=1e-5, dynamic_range=80.0, **kwargs):
        super(MagnitudeToDecibel, self).__init__(**kwargs)
        self.ref_value = ref_value
        self.amin = amin
        self.dynamic_range = dynamic_range

    def call(self, x):
        """
        Args:
            x (`Tensor`): float tensor. Can be batch or not. Something like magnitude of STFT.

        Returns:
            (`Tensor`): decibel-scaled float tensor of `x`.
        """
        return backend.magnitude_to_decibel(
            x, ref_value=self.ref_value, amin=self.amin, dynamic_range=self.dynamic_range
        )

    def get_config(self):
        config = super(MagnitudeToDecibel, self).get_config()
        config.update(
            {'amin': self.amin, 'dynamic_range': self.dynamic_range, 'ref_value': self.ref_value,}
        )
        return config


class ApplyFilterbank(Layer):
    """
    Apply a filterbank to the input spectrograms.


    Args:
        filterbank (`Tensor`): filterbank tensor in a shape of (n_freq, n_filterbanks)
        data_format (`str`): specifies the data format of batch input/output
        **kwargs: Keyword args for the parent keras layer (e.g., `name`)
    """

    def __init__(
        self, type, filterbank_kwargs, data_format='default', **kwargs,
    ):

        backend.validate_data_format_str(data_format)

        self.type = type
        self.filterbank_kwargs = filterbank_kwargs

        if type == 'log':
            self.filterbank = _log_filterbank = backend.filterbank_log(**filterbank_kwargs)
        elif type == 'mel':
            self.filterbank = _mel_filterbank = backend.filterbank_mel(**filterbank_kwargs)

        if data_format == CH_DEFAULT_STR:
            self.data_format = K.image_data_format()
        else:
            self.data_format = data_format

        if self.data_format == CH_FIRST_STR:
            self.freq_axis = 3
        else:
            self.freq_axis = 2
        super(ApplyFilterbank, self).__init__(**kwargs)

    def call(self, x):
        """
        Apply filterbank to `x`.

        Args:
            x (`Tensor`): float tensor in 2D batch shape.
        """

        # x: 2d batch input. (b, t, fr, ch) or (b, ch, t, fr)
        output = tf.tensordot(x, self.filterbank, axes=(self.freq_axis, 0))
        # ch_last -> (b, t, ch, new_fr). ch_first -> (b, ch, t, new_fr)
        if self.data_format == CH_LAST_STR:
            output = tf.transpose(output, (0, 1, 3, 2))
        return output

    def get_config(self):
        config = super(ApplyFilterbank, self).get_config()
        config.update(
            {
                'type': self.type,
                'filterbank_kwargs': self.filterbank_kwargs,
                'data_format': self.data_format,
            }
        )
        return config


class Delta(Layer):
    """Calculates delta, a local estimate of the derivative along time axis.
    See torchaudio.functional.compute_deltas or librosa.feature.delta for more details.

    Args:
        win_length (`int`): Window length of the derivative estimation. Defaults to 5
        mode (`str`): Specifies pad mode of `tf.pad`. Case-insensitive. Defaults to 'symmetric'.
            Can be 'symmetric', 'reflect', 'constant', or whatever `tf.pad` supports.

    """

    def __init__(self, win_length=5, mode='symmetric', data_format='default', **kwargs):
        backend.validate_data_format_str(data_format)

        if not win_length >= 3:
            raise ValueError(
                'win_length should be equal or bigger than 3, but it is %d' % win_length
            )
        if win_length % 2 != 1:
            raise ValueError('win_length should be an odd number, but it is %d' % win_length)
        if mode.lower() not in ('symmetric', 'reflect', 'constant'):
            raise ValueError(
                'mode.lower() should be one of {}'.format(str(('symmetric', 'reflect', 'constant')))
                + 'but it is {}'.format(mode)
            )

        if data_format == CH_DEFAULT_STR:
            self.data_format = K.image_data_format()
        else:
            self.data_format = data_format

        self.win_length = win_length
        self.mode = mode
        self.n = (self.win_length - 1) // 2  # half window length
        self.denom = 2 * sum([_n ** 2 for _n in range(1, self.n + 1, 1)])  # denominator
        super(Delta, self).__init__(**kwargs)

    def call(self, x):
        """
        Args:
            x (`Tensor`): a 2d batch (b, t, f, ch) or (b, ch, t, f)

        Returns:
            (`Tensor`): A tensor with the same shape as input data.

        """
        if self.data_format == 'channels_first':
            x = K.permute_dimensions(x, (0, 2, 3, 1))

        x = tf.pad(
            x, tf.constant([[0, 0], [self.n, self.n], [0, 0], [0, 0]]), mode=self.mode
        )  # pad over time
        kernel = K.arange(-self.n, self.n + 1, 1, dtype=K.floatx())
        kernel = K.reshape(kernel, (-1, 1, 1, 1))  # time, freq, in_ch, out_ch

        x = K.conv2d(x, kernel, data_format=CH_LAST_STR) / self.denom
        if self.data_format == CH_FIRST_STR:
            x = K.permute_dimensions(x, (0, 3, 1, 2))

        return x

    def get_config(self):
        config = super(Delta, self).get_config()
        config.update(
            {'win_length': self.win_length, 'mode': self.mode, 'data_format': self.data_format}
        )

        return config



class FrequencyAwareConv2D(Conv2D):
    """2D Frequency-Aware convolution layer (useful for Conv2D over spectograms).
    This layer has the same interface as Conv2D with one extra parameter
     indicating the frequency dimension
    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(300, 128, 1)?` for  300x128 spectograms with 1 channel
    in `data_format="channels_last"`.

    Examples:

    >>> # The inputs are 300x128 spectogram images with `channels_last` and the batch
    >>> # size is 4.
    >>> input_shape = (4, 300, 128, 1)
    >>> x = tf.random.normal(input_shape)
    >>> y = kapre.time_frequency.FreqAwareConv(
    ... 2, 3, activation='relu', input_shape=input_shape[1:])(x)
    >>> print(y.shape)
    (4, 300, 128, 2)
    References:
        Koutini, K., Eghbal-zadeh, H., & Widmer, G. (2019). Receptive-Field-Regularized CNN 
        Variants for Acoustic Scene Classification. In Proceedings of the Detection 
        and Classification of Acoustic Scenes and Events 2019 Workshop (DCASE2019).

        Liu, R., Lehman, J., Molino, P., Such, F. P., Frank, E., Sergeev, A., & Yosinski, J.
         (2018). An intriguing failing of convolutional neural networks and the coordconv
          solution. In Advances in Neural Information Processing Systems (pp. 9605-9616).

    Args:
        filters (int): the dimensionality of the output space (i.e. the number of
          output filters in the convolution).
        kernel_size (Union[int, tuple, list]): An integer or tuple/list of 2 integers, specifying the height
          and width of the 2D convolution window. Can be a single integer to specify
          the same value for all spatial dimensions.
        strides (Union[int, tuple, list], optional): An integer or tuple/list of 2 integers, specifying the strides of
          the convolution along the height and width. Can be a single integer to
          specify the same value for all spatial dimensions. Specifying any stride
          value != 1 is incompatible with specifying any `dilation_rate` value != 1.
        padding (str, optional): one of `"valid"` or `"same"` (case-insensitive).
          `"valid"` means no padding. `"same"` results in padding evenly to 
          the left/right or up/down of the input such that output has the same 
          height/width dimension as the input.
        data_format (str, optional): A string, one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs. `channels_last` corresponds
          to inputs with shape `(batch_size, height, width, channels)` while
          `channels_first` corresponds to inputs with shape `(batch_size, channels,
          height, width)`. It defaults to the `image_data_format` value found in
          your Keras config file at `~/.keras/keras.json`. If you never set it, then
          it will be `channels_last`.
        dilation_rate (Union[int, tuple, list], optional): specifying the
          dilation rate to use for dilated convolution. Can be a single integer to
          specify the same value for all spatial dimensions. Currently, specifying
          any `dilation_rate` value != 1 is incompatible with specifying any stride
          value != 1.
        groups (int, optional): A positive integer specifying the number of groups in which the
          input is split along the channel axis. Each group is convolved separately
          with `filters / groups` filters. The output is the concatenation of all
          the `groups` results along the channel axis. Input channels and `filters`
          must both be divisible by `groups`.
        activation (Union[str, :obj:`keras.activations`], optional): Activation function to use. If you don't specify anything, no
          activation is applied (see `keras.activations`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer (Union[str, :obj:`keras.initializers`], optional): Initializer for the `kernel` weights matrix (see
          `keras.initializers`).
        bias_initializer (Union[str, :obj:`keras.initializers`], optional): Initializer for the bias vector (see
          `keras.initializers`).
        kernel_regularizer (Union[str, :obj:`keras.regularizers`], optional): Regularizer function applied to the `kernel` weights
          matrix (see `keras.regularizers`).
        bias_regularizer (Union[str, :obj:`keras.regularizers`], optional): Regularizer function applied to the bias vector (see
          `keras.regularizers`).
        activity_regularizer (Union[str, :obj:`keras.regularizers`], optional): Regularizer function applied to the output of the
          layer (its "activation") (see `keras.regularizers`).
        kernel_constraint (Union[str, :obj:`keras.constraints`], optional): Constraint function applied to the kernel matrix (see
          `keras.constraints`).
        bias_constraint (Union[str, :obj:`keras.constraints`], optional): Constraint function applied to the bias vector (see
          `keras.constraints`).


    Input shape:
        4+D tensor with shape: `batch_shape + (channels, time, freq)` if
          `data_format='channels_first'`
        or 4+D tensor with shape: `batch_shape + (time, freq, channels)` if
          `data_format='channels_last'`.

    Output shape:
        4+D tensor with shape: `batch_shape + (filters, new_time, new_freq)` if
        `data_format='channels_first'` or 4+D tensor with shape: `batch_shape +
          (new_time, new_freq, filters)` if `data_format='channels_last'`.  `time`
          and `freq` values might have changed due to padding.

    Returns:
        A tensor of rank 4+ representing
        `activation(conv2d(inputs, kernel) + bias)`.

    Raises:
        ValueError: if `padding` is `"causal"`.
        ValueError: when both `strides > 1` and `dilation_rate > 1`.
        """
    def __init__(self,
               *args,
               **kwargs):
        if ('groups' in kwargs and kwargs.get('groups')>1 ) or \
             (len(args) >= 7 and args[7]>1):
            warnings.warn(
                'Grouped Conv2D are not supported.'
                'Only the first group will be frequency aware.'
            )
        super(FrequencyAwareConv2D, self).__init__(*args,
            **kwargs)


    def call(self, inputs):
        inputs = self._add_freq_info_channel(inputs)
        outputs = super(FrequencyAwareConv2D, self).call(inputs)
        return outputs

    def _add_freq_info_channel(self, inputs):
        shape = tf.shape(inputs)
        time_axis, freq_axis, ch_axis = (1, 2, 3) if self.data_format == 'channels_last' \
            else (2, 3, 1)
        n_batch, n_freq, n_time, n_ch = shape[0], shape[freq_axis], shape[time_axis], shape[ch_axis]
        # freq_info shape: n_freq
        freq_info=tf.cast(tf.range(n_freq) , tf.float32) / tf.cast(n_freq - 1, tf.float32)
        # repeate over time, shape: n_time,n_freq
        time_freq_info=tf.tile(tf.expand_dims(freq_info, 0),[n_time,1])
        # expand to a channels dim (n_time,n_freq,1)
        time_freq_channel_info=tf.expand_dims(time_freq_info, ch_axis-1)
        # repeate over batch (n_time,n_freq,1)
        batch_time_freq_channel_info=tf.tile(tf.expand_dims(time_freq_channel_info, 0),
            [n_batch,1,1,1])
        return  tf.concat([batch_time_freq_channel_info, inputs], axis=ch_axis)

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
          raise ValueError('The channel dimension of the inputs '
                           'should be defined. Found `None`.')
        original_input_channels=int(input_shape[channel_axis])
        return original_input_channels+1







class ConcatenateFrequencyMap(Layer):
    """Addes a frequency information channel to the 4D input feature map.
    References:
        Koutini, K., Eghbal-zadeh, H., & Widmer, G. (2019). Receptive-Field-Regularized CNN 
        Variants for Acoustic Scene Classification. In Proceedings of the Detection 
        and Classification of Acoustic Scenes and Events 2019 Workshop (DCASE2019).

        Liu, R., Lehman, J., Molino, P., Such, F. P., Frank, E., Sergeev, A., & Yosinski, J.
         (2018). An intriguing failing of convolutional neural networks and the coordconv
          solution. In Advances in Neural Information Processing Systems (pp. 9605-9616).

    Args:
        data_format (str, optional): A string, one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs. `channels_last` corresponds
          to inputs with shape `(batch_size, height, width, channels)` while
          `channels_first` corresponds to inputs with shape `(batch_size, channels,
          height, width)`. It defaults to the `image_data_format` value found in
          your Keras config file at `~/.keras/keras.json`. If you never set it, then
          it will be `channels_last`.

    """

    def __init__(self,data_format= 'channels_last',**kwargs):
        super(ConcatenateFrequencyMap, self).__init__(**kwargs)
        self.data_format = 'channels_last'

    def call(self, inputs):
        return self._add_freq_info_channel(inputs)
    def get_config(self):
        config = super(ConcatenateFrequencyMap, self).get_config()
        config.update(
            {
                'data_format': self.data_format,
            }
        )
        return config
    def _add_freq_info_channel(self, inputs):
        shape = tf.shape(inputs)
        time_axis, freq_axis, ch_axis = (1, 2, 3) if self.data_format == 'channels_last' \
            else (2, 3, 1)
        n_batch, n_freq, n_time, n_ch = shape[0], shape[freq_axis], shape[time_axis], shape[ch_axis]
        # freq_info shape: n_freq
        freq_info=tf.cast(tf.range(n_freq) , tf.float32) / tf.cast(n_freq - 1, tf.float32)
        # repeate over time, shape: n_time,n_freq
        time_freq_info=tf.tile(tf.expand_dims(freq_info, 0),[n_time,1])
        # expand to a channels dim (n_time,n_freq,1)
        time_freq_channel_info=tf.expand_dims(time_freq_info, ch_axis-1)
        # repeate over batch (n_time,n_freq,1)
        batch_time_freq_channel_info=tf.tile(tf.expand_dims(time_freq_channel_info, 0),
            [n_batch,1,1,1])
        return  tf.concat([batch_time_freq_channel_info, inputs], axis=ch_axis)