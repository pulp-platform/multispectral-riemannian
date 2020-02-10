""" Contains some utility functions """

__author__ = "Tibor Schneider"
__email__ = "sctibor@student.ethz.ch"

import numpy as np

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
