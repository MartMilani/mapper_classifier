
__version__ = '0.1'
__date__ = 'January 28, 2019'

import sys
if sys.hexversion < 0x03000000:
    raise ImportError('PredMap requires at least Python version 3.0.')
del sys

from predmap._predmap import *
