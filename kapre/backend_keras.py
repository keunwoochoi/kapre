# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
from keras import backend as K


def amplitude_to_decibel(x, amin=1e-10, dynamic_range=80.0):
    """[K] Convert (linear) amplitude to decibel (log10(x)).

    x: Keras tensor or variable. 

    amin: minimum amplitude. amplitude smaller than `amin` is set to this.

    dynamic_range: dynamic_range in decibel
    """
    log_spec = 10 * K.log(K.maximum(x, amin)) / np.log(10).astype(K.floatx())
    log_spec = log_spec - K.max(log_spec)  # [-?, 0]
    log_spec = K.maximum(log_spec, -1 * dynamic_range)  # [-80, 0]
    return log_spec
