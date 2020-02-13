""" Contains some utility functions """

__author__ = "Tibor Schneider"
__email__ = "sctibor@student.ethz.ch"

import numpy as np
import unittest

def quantize(data, factor, num_bits, do_round=False):
    """ Quantize the data to the given number of levels """

    max_val = 1 << (num_bits - 1)
    data = data / factor
    data = data * max_val
    data = np.clip(data, -max_val, max_val - 1)
    if do_round:
        data = data.round()
    else:
        data = (data.astype(int)).astype(float)
    data = data / max_val
    data = data * factor
    return data


def quantize_to_int(data, factor, num_bits, do_round=False):
    """ Quantizes the data and returns an integer array """
    max_val = 1 << (num_bits - 1)
    data = data / factor
    data = data * max_val
    data = np.clip(data, -max_val, max_val - 1)
    if do_round:
        data = data.round()
    data = data.astype(int)
    return data


def dequantize_to_float(data, factor, num_bits):
    """ De-Quantizes the data and returns an float array """
    max_val = 1 << (num_bits - 1)
    data = data.astype(float)
    data = data / max_val
    data = data * factor
    return data
