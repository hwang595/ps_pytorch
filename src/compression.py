import numpy as np
import blosc

import time
from sys import getsizeof

'''we use different strategies for gradient compression and layer weight compression given API of mpi4py'''
def _trim_msg(msg):
    """
    msg : bytearray
        Somewhere in msg, 32 elements are 0x29. Returns the msg before that
    """
    i = msg.find(b'\x29'*32)
    if i == -1:
        raise Exception('trim_msg error; end of msg not found')
    return msg[:i]

def g_compress(grad):
    assert isinstance(grad, np.ndarray)
    compressed_grad = blosc.pack_array(grad, cname='snappy')
    return compressed_grad

def g_decompress(msg):
    assert isinstance(msg, str)
    grad = blosc.unpack_array(msg)
    return grad

def w_compress(w):
    assert isinstance(w, np.ndarray)
    msg = w.tobytes()
    packed_msg = blosc.compress(msg, cname='snappy')
    # add redundency to avoid blosc decompress issue
    send_msg = packed_msg + bytearray(b'\x29'*32)
    return send_msg

def w_decompress(msg,shape):
    trimmed_msg = trim_msg(msg)
    unpacked_weight = blosc.decompress(trimmed_msg)
    weight = np.fromstring(unpacked_weight).reshape(shape)
    return weight