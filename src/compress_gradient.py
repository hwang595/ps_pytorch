import numpy as np
import blosc

import time
from sys import getsizeof

def compress(grad):
	assert isinstance(grad, np.ndarray)
	compressed_grad = blosc.pack_array(grad, cname='snappy')
	return compressed_grad

def decompress(msg):
	assert isinstance(msg, str)
	grad = blosc.unpack_array(msg)
	return grad