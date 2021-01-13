# -*- coding: utf-8 -*-
# workarounds for missing TFLite support for rfft and stft and tf.signal.frame
# improved on and take from:
# https://github.com/tensorflow/magenta/tree/master/magenta/music
# as posted in https://github.com/tensorflow/tensorflow/issues/27303
# and https://gist.github.com/padoremu/8288b47ce76e9530eb288d4eec2e0b4d

import math

import tensorflow as tf
import numpy as np


def _rdft_matrix(dft_length):
    """Return a precalculated DFT matrix for positive frequencies only
    Args:
        dft_length (int) - DFT length
    Returns
        rdft_mat (k x n tensor) - precalculated DFT matrix rows are frequencies
            columns are samples, k dimension is dft_length // 2 +1 bins long
    """
    # freq bins
    k = np.arange(0, dft_length // 2 + 1)
    # Samples
    n = np.arange(0, dft_length)
    # complex frequency vector (now normalised to 2 pi)
    omega = -1j * 2.0 * np.pi / dft_length * k
    # complex phase, compute a matrix of value for the complex phase for each sample
    # location (n) and each freq bin (k) outer product If the two vectors have dimensions
    # k and n, then their outer product is an k × n matrix
    phase = np.outer(omega, n)
    # return transposed ready for matrix multiplication
    return np.exp(phase).astype(np.complex64).T


@tf.function
def _rdft(signal, dft_length):
    """DFT for real signals.
    Calculates the onesided dft, assuming real signal implies complex congugaqe symetry,
    hence only onesided DFT is returned.
    Args:
        signal (tensor) signal to transform, assumes that the last dimension is the time dimension
            signal can be framed, e.g. (1, 40, 1024) for a single batch of 40 frames of
            length 1024
        dft_length (int) - DFT length
    Returns:
        spec_real (float32 tensor) - real part of spectrogram, e.g. (1, 40, 513) for a 1024 length dft
        spec_imag (float32 tensor) - imag part of spectrogram, e.g. (1, 40, 513) for a 1024 length dft
    """
    # calculate the positive frequency atoms, and tell tensorflow this is a constant.
    rdft_mat = _rdft_matrix(dft_length)

    # tflite doest support complex types so split into real and imaginary:
    rdft_mat_real = tf.constant(np.real(rdft_mat))
    rdft_mat_imag = tf.constant(np.imag(rdft_mat))

    # Center-padding, in case the frame length and DFT lenght are different,
    # pad the signal to the center of the frame
    frame_length = tf.shape(signal)[-1]
    half_pad = (dft_length - frame_length) // 2
    pad_values = tf.concat(
        [tf.zeros([tf.rank(signal) - 1, 2], tf.int32), [[half_pad, half_pad]]], axis=0
    )
    signal_padded = tf.pad(signal, pad_values)

    # matrix multiplying real and imag seperately is faster than using complex types.
    spec_real = tf.matmul(signal_padded, rdft_mat_real)
    spec_imag = tf.matmul(signal_padded, rdft_mat_imag)
    spectrogram = tf.stack([spec_real, spec_imag], axis=-1)

    return spectrogram


def fixed_frame(signal, frame_length, frame_step):
    """tflite-compatible tf.signal.frame for fixed-size input.
    Args:
        signal: Tensor containing signal(s).
        frame_length: Number of samples to put in each frame.
        frame_step: Sample advance between successive frames.
    Returns:
        A new tensor where the last axis (or first, if first_axis) of input
        signal has been replaced by a (num_frames, frame_length) array of individual
        frames where each frame is drawn frame_step samples after the previous one.
    Raises:
        ValueError: if signal has an undefined axis length.  This routine only
        supports framing of signals whose shape is fixed at graph-build time.
    """
    signal_shape = list(signal.shape)
    length_samples = signal_shape[-1]

    if length_samples <= 0:
        raise ValueError("fixed framing requires predefined constant signal length")
    # the number of whole frames
    num_frames = max(0, 1 + (length_samples - frame_length) // frame_step)

    # define the output_shape, if we recieve a None dimension, replace with 1
    outer_dimensions = [dim if dim else 1 for dim in signal_shape[:-1]]
    # outer_dimensions = signal_shape[:-1]
    output_shape = outer_dimensions + [num_frames, frame_length]

    # Currently tflite's gather only supports axis==0, but that may still
    # work if we want the last of 1 axes.
    gather_axis = len(outer_dimensions)

    # subframe length is the largest int that as a common divisor of the frame
    # length and hop length. We will slice the signal up into these subframes
    # in order to then construct the frames.
    subframe_length = math.gcd(frame_length, frame_step)
    subframes_per_frame = frame_length // subframe_length
    subframes_per_hop = frame_step // subframe_length
    num_subframes = length_samples // subframe_length

    # define the subframe shape and the trimmed audio length, removeing any unused
    # excess audio, so subframe fit exactly.
    subframe_shape = outer_dimensions + [num_subframes, subframe_length]
    trimmed_input_size = outer_dimensions + [num_subframes * subframe_length]
    # slice up the audio into subframes
    subframes = tf.reshape(
        tf.slice(signal, begin=np.zeros(len(signal_shape), np.int32), size=trimmed_input_size),
        subframe_shape,
    )

    # frame_selector is a [num_frames, subframes_per_frame] tensor
    # that indexes into the appropriate frame in subframes. For example:
    # [[0, 0, 0, 0], [2, 2, 2, 2], [4, 4, 4, 4]]
    frame_selector = np.reshape(np.arange(num_frames) * subframes_per_hop, [num_frames, 1])

    # subframe_selector is a [num_frames, subframes_per_frame] tensor
    # that indexes into the appropriate subframe within a frame. For example:
    # [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    subframe_selector = np.reshape(np.arange(subframes_per_frame), [1, subframes_per_frame])

    # Adding the 2 selector tensors together produces a [num_frames,
    # subframes_per_frame] tensor of indices to use with tf.gather to select
    # subframes from subframes. We then reshape the inner-most subframes_per_frame
    # dimension to stitch the subframes together into frames. For example:
    # [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]].
    selector = frame_selector + subframe_selector
    frames = tf.reshape(
        tf.gather(subframes, selector.astype(np.int32), axis=gather_axis), output_shape
    )

    return frames


def stft_tflite(signal, frame_length, frame_step, fft_length, window_fn, pad_end):
    """tflite-compatible implementation of tf.signal.stft.
    Compute the short-time Fourier transform of a 1D input while avoiding tf ops
    that are not currently supported in tflite (Rfft, Range, SplitV).
    fft_length must be fixed. A Hann window is of frame_length is always
    applied.
    Since fixed (precomputed) framing must be used, signal.shape[-1] must be a
    specific value (so "?"/None is not supported).
    Args:
        signal: 1D tensor containing the time-domain waveform to be transformed.
        frame_length: int, the number of points in each Fourier frame.
        frame_step: int, the number of samples to advance between successive frames.
        fft_length: int, the size of the Fourier transform to apply.
    Returns:
        Two (num_frames, fft_length) tensors containing the real and imaginary parts
        of the short-time Fourier transform of the input signal.
    """
    signal = tf.cast(signal, tf.float32)
    if pad_end:
        # the number of whole frames
        length_samples = signal.shape[-1]
        num_steps_round_up = tf.math.ceil(length_samples / frame_step)
        pad_amount = int((num_steps_round_up * frame_step) - length_samples)
        signal = tf.pad(signal, tf.constant([[0, 0], [0, 0], [0, pad_amount]]))
    # Make the window be shape (1, frame_length) instead of just frame_length
    # in an effort to help the tflite broadcast logic.
    window = tf.reshape(window_fn(frame_length), [1, frame_length])

    framed_signal = fixed_frame(signal, frame_length, frame_step)
    framed_signal *= window

    spectrogram = _rdft(framed_signal, fft_length)

    return spectrogram
