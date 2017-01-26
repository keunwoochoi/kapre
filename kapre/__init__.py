__version__ = '0.0.3'
VERSION = __version__

from . import time_frequency
from . import backend
from . import backend_keras

from . import augmentation
from . import filterbank
from . import utils

from keras import backend as K

if K.backend() == 'theano':
    try:
        from theano.tensor import fft
        from . import stft
    except:
        print('Update theano to 0.9 to use stft.')
