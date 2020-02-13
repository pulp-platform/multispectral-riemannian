""" Contains some functions for computing quantized SOS IIR filters """

__author__ = "Tibor Schneider"
__email__ = "sctibor@student.ethz.ch"

import numpy as np
import unittest

from utils import quantize_to_int, dequantize_to_float, quantize


N_FILTER_BITS = 10


def prepare_quant_filter(coeff, x_scale, y_scale, n_bits=N_FILTER_BITS):
    """ Quantizes the sos filter coefficients and prepares the scale ranges.

    Description
    -----------
    All filter coefficients will have a scale factor which is a power of two. This way, we can
    implement the rescaling as bitshifts.

    We define different scale regions. We use the direct form 2 of the IIR filter
    (see https://en.wikipedia.org/wiki/Digital_biquad_filter#Direct_form_2). A new scale region
    is defined after the numerator MACs (b_k). To compute the new scale region, we use the
    scale_coeff at the specific section. By assuming that we still multiply 8 bit values, we just
    shift by 8 bit. The scale factor at the output of section m thus becomes the product of all
    numerator scale factors from 0 up to, and including, m.

        a_shift[m] = log2(a_scale[m]) - (n_bits - 1)
        b_shift[m] =                  - (n_bits - 1)
        y_shift    = log2(x_scale / y_scale) + sum_m log2(b_scale[m])

    Parameters
    ----------

    coeff: np.array, size=(M, 6)
           filter coefficients, with M second order sections. The first three represent the
           numerator, and the second the denominator.

    x_scale: float
             Scale factor for the input

    y_scale: float
             Scale factor for the output, must be a power of two of the input scale

    n_bits: int
            Number of bits used to represent filter values

    Returns
    -------

    quant_coeff: np.array, size=(M, 6)
                 Quantized coefficients, but still in a floating point representation

    scale_coeff: np.array, size=(M, 2)
                 Scale factor as a power of two for b_scale: (M, 0) and a_scale: (M, 1)

    comp_shift: np.array, size=(M, 2)
                Bitshift (to the right) to apply for computing, b_shift: (M, 0) and a_shift: (M, 1)

    y_shift: int:
             Bitshift (to the right) to apply after the last section, for converting back to the
             output scale
    """

    # check the data
    assert len(coeff.shape) == 2
    assert coeff.shape[1] == 6

    M = coeff.shape[0]

    quant_coeff = np.zeros_like(coeff)
    scale_coeff = np.zeros((M, 2))
    comp_shift = np.zeros((M, 2))
    y_shift = int(np.round(np.log2(x_scale / y_scale)))

    for m in range(M):
        # compute the scale and the shift
        a_scale = np.abs(coeff[m, 3:]).max()
        a_shift = int(np.ceil(np.log2(a_scale)))
        a_scale = 2 ** a_shift
        b_scale = np.abs(coeff[m, :3]).max()
        b_shift = int(np.ceil(np.log2(b_scale)))
        b_scale = 2 ** b_shift

        # update the output shift now
        y_shift += b_shift

        # modify the shift for denominators (a), they must be in the same range as the section input
        a_shift = a_shift - (n_bits - 1)

        # set the shift for the numerators (b), they get their new range
        b_shift = -(n_bits - 1)

        # store the values
        scale_coeff[m, 1] = a_scale
        scale_coeff[m, 0] = b_scale
        comp_shift[m, 1] = a_shift
        comp_shift[m, 0] = b_shift

        # quantize the coefficients
        quant_coeff[m, 3:] = quantize(coeff[m, 3:], a_scale, num_bits=n_bits, do_round=True)
        quant_coeff[m, :3] = quantize(coeff[m, :3], b_scale, num_bits=n_bits, do_round=True)

    return quant_coeff, scale_coeff, comp_shift.astype(int), y_shift


def quant_sos_filt(data, quant_filter, scale_data, scale_output, mode="sosfilt", n_bits=8,
                   intermediate_bits=32, filter_bits=N_FILTER_BITS):
    """ applies sos filter in quantized form (Direct Form 2)

    Description
    -----------
    We try to fit all intermediate values in 8 bits. However, the numbers are represented as 16 bit
    values to make sure they do not overflow. Also, by doing it like this, we can still use SIMD
    operations in the Mr. Wolf implementation.

    To achieve this goal, we define different scale regions. We use the direct form 2 of the IIR
    filter (see https://en.wikipedia.org/wiki/Digital_biquad_filter#Direct_form_2). A new scale
    region is defined after the numerator MACs (b_k). To compute the new scale region, we use the
    scale_coeff at the specific section. By assuming that we still multiply 8 bit values, we just
    shift by 8 bit. The scale factor at the output of section m thus becomes the product of all
    numerator scale factors from 0 up to, and including, m.

    Actually, since we are working with signed 8 bit values, the range is just 2^7, thus we need to
    shift by 7 bits instead of 8.

    Parameters
    ----------

    data: np.array, size=N
          Input signal

    quant_filter: tuple (quant_coeff, scale_coeff, comp_shift, scale_output):

        quant_coeff: np.array, size=(M, 6)
                     filter coefficients, with M second order sections. The first three represent
                     the numerator, and the second the denominator.

        scale_coeff: np.array, size=(M, 2)
                     Scale factors for the coefficients, where the first represents the scale factor
                     for the numerator and the second for the denominator.

        comp_shift: np.array, size=(M, 2), dtype=int
                    amount to shift during computation, for all sections, numerator and denominators

        output_shift: int
                      amount to shift the output data (to the left)

    scale_data: float
                Scale factor for the input data

    scale_output: float
                  Scale factor of the output data, must be a power of two

    mode: str { "valid", "same", "full", "sosfilt" }
          Mode for truncating the output. The mode "sosfilt" behaves the like scipy.signal.sosfilt

    n_bits: int
            Number of bits for the computation, must be equal to 8

    intermediate_bits: int
                       Number of bits to represent the intermediate values

    filter_bits: int
                 Number of bits to represent the filter coefficients

    Returns:
    --------

    np.array, size=N': result
    """

    coeff, scale_coeff, comp_shift, y_shift = quant_filter

    # check data
    assert len(data.shape) == 1
    assert len(coeff.shape) == 2
    assert coeff.shape[1] == 6
    assert scale_coeff.shape[0] == coeff.shape[0]
    assert scale_coeff.shape[1] == 2
    assert mode in ["valid", "same", "full", "sosfilt"]
    assert n_bits == 8

    N = data.shape[0]
    M = coeff.shape[0]
    N_prime = N + M * 2 # per section, two samples are added

    # quantize x
    x_quant = quantize_to_int(data, scale_data, n_bits)

    # generate the coefficients
    a = np.ones((M + 1, 3), dtype=int) << (n_bits - 1)
    b = np.ones((M + 1, 3), dtype=int) << (n_bits - 1)
    a_shift = np.ones((M + 1, ), dtype=int) * (n_bits - 1)
    b_shift = np.ones((M + 1, ), dtype=int) * (n_bits - 1)

    for m in range(M):
        a[m + 1, :] = quantize_to_int(coeff[m, 3:], scale_coeff[m, 1], filter_bits)
        b[m + 1, :] = quantize_to_int(coeff[m, :3], scale_coeff[m, 0], filter_bits)
        a_shift[m + 1] = -comp_shift[m, 1]
        b_shift[m + 1] = -comp_shift[m, 0]

    # values for storing the data.
    regs = np.zeros((M + 1, 3), dtype=int)
    y = np.zeros((N_prime, ), dtype=int)

    # pad x such that we do not get any index errors
    x = np.zeros((N_prime, ), dtype=int)
    x[:N] = x_quant

    # internal representation has n_bits: n_bits * 2 (because we multiply two such numbers together)
    # x = x << (n_bits - 1)

    # main body of the computation
    for k in range(N_prime):
        # get the new element
        regs[0, 0] = x[k]

        # compute all the sections
        for m in range(1, M + 1):
            # move the registers
            regs[m, 2] = regs[m, 1]
            regs[m, 1] = regs[m, 0]
            regs[m, 0] = 0

            # add the input from the previous section
            regs[m, 0] += np.sum(regs[m - 1, :] * b[m - 1, :]) >> b_shift[m - 1]

            # add the self reference
            regs[m, 0] -= np.sum(regs[m, 1:] * a[m, 1:]) >> a_shift[m]

            # clip the values
            regs[m] = np.clip(-(1 << (intermediate_bits - 1)),
                              1 << (intermediate_bits - 1),
                              regs[m])

        # store the output
        y[k] = np.sum(regs[M, :] * b[M, :]) >> b_shift[M]

    # apply the mode
    if mode == "same":
        y = y[M:-M]
    if mode == "valid":
        y = y[2 * M:-2 * M]
    if mode == "sosfilt":
        y = y[:N]

    # shift y back by the amount necessary
    if y_shift < 0:
        y = y >> -y_shift
    elif y_shift > 0:
        y = y << y_shift

    # dequantize to float
    result = dequantize_to_float(y, scale_output, n_bits)

    return result


def _plot(band_id, coeff):
    from scipy.signal import sosfilt
    import matplotlib.pyplot as plt


    for w in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
        #x = np.random.randn(825)
        x = np.sin(w * np.array(range(1000)))
        y_opt = sosfilt(coeff, x)

        x_scale = np.abs(x).max()
        y_scale = np.abs(y_opt).max()
        k = np.ceil(np.log2(y_scale / x_scale))
        y_scale = x_scale * (2 ** k)
        quant_filter = prepare_quant_filter(coeff, x_scale, y_scale)

        y_exp = sosfilt(quant_filter[0], x)
        y_acq = quant_sos_filt(x, quant_filter, x_scale, y_scale)

        plt.plot(y_exp, label="expected")
        plt.plot(y_acq, label="acquired")
        plt.plot(y_opt, label="optimal")
        plt.legend()
        plt.show()


def _sweep(band_id, coeff, freqs=None, N=500, T=1000, fs=250):
    from scipy.signal import sosfilt
    import matplotlib.pyplot as plt
    from p_tqdm import p_map

    if freqs is None:
        freqs = np.linspace(0, np.pi / 2, N)

    t = np.array(range(T))

    def measure(w):
        x = np.cos(w * t)

        y_exp_f = sosfilt(coeff, x)

        y_scale = np.abs(y_exp_f).max()
        y_scale = 2 ** np.ceil(np.log2(y_scale))

        quant_filter = prepare_quant_filter(coeff, 1, y_scale)

        y_acq = quant_sos_filt(x, quant_filter, 1, y_scale)
        y_exp_q = sosfilt(quant_filter[0], x)

        a_acq = np.abs(y_acq[-100:]).max()
        a_exp_f = np.abs(y_exp_f[-100:]).max()
        a_exp_q = np.abs(y_exp_q[-100:]).max()

        return np.array([a_acq, a_exp_f, a_exp_q])

    measurement = p_map(measure, list(freqs), desc=f"Band {band_id}")
    measurement = np.array(measurement)
    ampl_acq = measurement[:, 0]
    ampl_exp_f = measurement[:, 1]
    ampl_exp_q = measurement[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(freqs, ampl_acq, label="fully quantized")
    ax.plot(freqs, ampl_exp_q, label="quantized weights")
    ax.plot(freqs, ampl_exp_f, label="folat weights")
    ax.legend()
    ax.set_yscale('log')
    plt.show()

def _test():
    """ test function to run the frequency sweep on all filterbands """
    from filters import load_filterbank
    bank = load_filterbank([2], fs=250, order=2)
    for i, coeff in enumerate(bank):
        #_plot(i, coeff)
        _sweep(i, coeff)

if __name__ == "__main__":
    _test()
