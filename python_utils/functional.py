"""
This file contains functions for (Quantized) Neural Networks
"""

__author__ = "Tibor Schneider"
__email__ = "sctibor@student.ethz.ch"
__version__ = "0.1.0"
__date__ = "2020/01/23"

import numpy as np


def quantize(x, scale_factor, n_bits=8, do_round=True):
    """
    Quantizes the input linearly (without offset) with the given number of levels.
    The quantization levels will be: 
        np.linspace(-scale_factor, scale_facotr, num_levels)
    The output will contain only quantized values (not the integer representation)

    Parameters:
    - x: np.array(dtype=float), original vector
    - scale_factor: float, the output will be quantized to range [-s, s]
    - num_levels: int, number of bits for the quantization
    - do_round: bool, if True, round to the nearest integer. If false, round towards zero

    Returns: np.array(dtype=float), where all values are within the quantized grid
    """
    x_q = quantize_to_int(x, scale_factor, n_bits, do_round)
    return dequantize(x_q, scale_factor, n_bits)


def quantize_to_int(x, scale_factor, n_bits=8, do_round=True):
    """
    Quantizes the input linearly (without offset) with the given number of levels.
    The quantization levels will be:
        np.linspace(-scale_factor, scale_facotr, num_levels)
    The output values will be one of:
        [-(1<<(n_bits-1)), ..., -1, 0, 1, ..., (1<<(n_bits-1))-1]
    As an example, n_bits = 8, we have a range
        [-128, -126, ..., -1, 0, 1, ..., 126, 127]
    The value will be floored towards zero, just like integer division in C

    Parameters:
    - x: np.array(dtype=float), original vector
    - scale_factor: float, the output will be quantized to range [-s, s]
    - num_levels: int, number of quantization levels, must be odd

    Returns: np.array(dtype=int), where all values will be in the integer representation
    """

    min_level = -(1 << (n_bits - 1))
    max_level = (1 << (n_bits - 1)) - 1
    integer_range = 1 << (n_bits - 1)

    x = x / scale_factor
    x = x * integer_range
    x = np.clip(x, min_level, max_level)
    if do_round:
        x = np.round(x)
    x = x.astype(np.int)
    return x


def dequantize(x, scale_factor, n_bits=8):
    """
    Reverse operation of quantize_to_int

    Parameters:
    - x: np.array(dtype=int), quantized vector in integer representation
    - scale factor: float input will be mapped to this range
    - n_bits: number of bits for the representation

    Returns: np.array(dtype=float), in float representation
    """
    min_level = -(1 << (n_bits - 1))
    max_level = (1 << (n_bits - 1)) - 1
    integer_range = 1 << (n_bits - 1)

    # check for overflow
    if x.min() < min_level or x.max() > max_level:
        raise OverflowError()

    x = x / integer_range
    x = x * scale_factor
    return x


def quantize_iir_filter(filter_dict, n_bits):
    """
    Quantize the iir filter tuple for sos_filt funcitons

    Parameters:
    - filter_dict: dict, contains the quantized filter dictionary with the following keys:
      - coeff: np.array(size=(M, 6)), float representation of the coefficients
      - coeff_scale: np.array(size=(M, 2)), scale all coefficients, not used here
      - coeff_shift: np.array(size=(M, 2), dtype=int), amount to shift during computation
      - y_scale: float, scale factor of the output, unused here
      - y_shift: int, number of bits to shift the output for scaling
    - n_bits: int, number of bits to represent the filter coefficients

    Returns: tuple:
    - a: np.array(size=(M+1, 3), dtype=int), quantized nominators
    - a_shift: np.array(size=(M+1), dtype=int), amount to shift during computation
    - b: np.array(size=(M+1, 3), dtype=int), quantized denumerators
    - b_shift: np.array(size=(M+1), dtype=int), amount to shift during computation
    - y_shift: int, amount to shift the output
    """

    quant_coeff = filter_dict["coeff"]
    scale_coeff = filter_dict["coeff_scale"]
    comp_shift = filter_dict["coeff_shift"]
    output_shift = filter_dict["y_shift"]

    M = quant_coeff.shape[0]
    assert quant_coeff.shape == (M, 6)
    assert scale_coeff.shape == (M, 2)
    assert comp_shift.shape == (M, 2)
    assert comp_shift.dtype == int
    assert np.all(comp_shift <= 0)

    # generate the coefficients
    a = np.ones((M + 1, 3), dtype=int) << (n_bits - 1)
    b = np.ones((M + 1, 3), dtype=int) << (n_bits - 1)
    a_shift = np.ones((M + 1, ), dtype=int) * (n_bits - 1)
    b_shift = np.ones((M + 1, ), dtype=int) * (n_bits - 1)

    for m in range(M):
        a[m + 1, :] = quantize_to_int(quant_coeff[m, 3:], scale_coeff[m, 1], n_bits)
        b[m + 1, :] = quantize_to_int(quant_coeff[m, :3], scale_coeff[m, 0], n_bits)
        a_shift[m + 1] = -comp_shift[m, 1]
        b_shift[m + 1] = -comp_shift[m, 0]

    return a, a_shift, b, b_shift, output_shift


def prepare_bitshift(a_scale, a_n_bits, b_scale, b_n_bits, y_scale, y_n_bits):
    """ Computes the required bitshift to achieve the desired scaling.

    In the full precision model, assume the equation y = a * b. Then, assume that a, b and y are
    represented as fix point representation: a' = Ra / Sa, b' = Rb / Sb, y' = Ry / Sy, where:

        Rx = 1 << (Nx - 1), where Nx is the number of bits for the quantized vairable x (a, b or y).
    
    The resulting bitshift (to the left) is computed as:

              Ra * Rb * Sy
        log2( ------------ )
              Sa * Sb * Ry

    Parameters:
    - a_scale: float, Sa
    - a_n_bits: int, Ra = 1 << (a_n_bits - 1)
    - b_scale: float, Sb
    - b_n_bits: int, Rb = 1 << (b_n_bits - 1)
    - y_scale: float, Sy
    - y_n_bits: int, Ry = 1 << (y_n_bits - 1)

    Returns: int, bitshift to scale after multiplication a * b, bitshift is always to the right
    """

    a_range = 1 << (a_n_bits - 1)
    b_range = 1 << (b_n_bits - 1)
    y_range = 1 << (y_n_bits - 1)

    factor = (a_range * b_range * y_scale) / (a_scale * b_scale * y_range)

    # the factor must be really close to a power of two
    np.testing.assert_almost_equal(np.log2(factor), round(np.log2(factor)))

    bitshift = int(np.round(np.log2(factor)))
    return bitshift


def solve_for_scale_sqr(a_scale, a_n_bits, b_scale, b_n_bits, y_n_bits):
    """ Finds the scale factor for the following case:

        y = a * b^2

    We know the scale factor and the number of bits of a, as well as b. We also know the number of
    bits for y. This function computes the resulting scale factor for y

    Parameters:
    - a_scale: float
    - a_n_bits: int
    - b_scale: float
    - b_n_bits: int
    - y_n_bits: int

    Returns: float, resulting scale factor for y, can be used for dequantization
    """
    a_range = (1 << (a_n_bits - 1))
    b_range = (1 << (b_n_bits - 1))
    y_range = (1 << (y_n_bits - 1))

    return (y_range * a_scale * b_scale * b_scale) / (a_range * b_range * b_range)


def solve_for_scale(a_scale, a_n_bits, b_scale, b_n_bits, y_n_bits):
    """ Finds the scale factor for the following case:

        y = a * b

    We know the scale factor and the number of bits of a, as well as b. We also know the number of
    bits for y. This function computes the resulting scale factor for y

    Parameters:
    - a_scale: float
    - a_n_bits: int
    - b_scale: float
    - b_n_bits: int
    - y_n_bits: int

    Returns: float, resulting scale factor for y, can be used for dequantization
    """
    a_range = (1 << (a_n_bits - 1))
    b_range = (1 << (b_n_bits - 1))
    y_range = (1 << (y_n_bits - 1))

    return (y_range * a_scale * b_scale) / (a_range * b_range)


def apply_bitshift_scale(x, bitshift, do_round=True):
    """ applies bitshift scale to the vector x

    Parameters:
    - x: np.array(dtype=int), input array
    - bitshift: int: bitshift to the right
    - do_round: add the rounding factor if this is enabled

    Returns: np.array(dtype=int), scaled vector
    """

    assert bitshift >= 0

    if do_round and bitshift > 0:
        #x += np.sign(x) * (1 << (bitshift - 1))
        x += (1 << (bitshift - 1))

    # check if an overflow is happending
    if x.min() < -(1 << 31) or x.max() > ((1 << 31) - 1):
        raise OverflowError()

    x = x >> bitshift

    return x
