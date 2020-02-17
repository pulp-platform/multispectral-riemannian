""" Contains some functions for computing quantized SOS IIR filters """

__author__ = "Tibor Schneider"
__email__ = "sctibor@student.ethz.ch"

import numpy as np

from utils import quantize_to_int, dequantize_to_float, quantize

# used for testing
from scipy.signal import sosfilt
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
from filters import load_filterbank, load_bands
from functools import partial

N_FILTER_BITS = 12
BIT_RESERVE = 0
INTERMEDIATE_BITS = 16
DEFAULT_FORM = 1
OVERFLOW_WARNING = True
OVERFLOW_RAISE = False

def prepare_quant_filter(coeff, x_scale, y_scale, n_bits=N_FILTER_BITS, bit_reserve=BIT_RESERVE):
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

    bit_reserve: int
                 Number of bits to shift the registers, may help when the registers are clipping,
                 but reduces the accuracy.

    Returns
    -------

    quant_coeff: np.array, size=(M, 6)
                 Quantized coefficients, but still in a floating point representation

    scale_coeff: np.array, size=(M, 2)
                 Scale factor as a power of two for b_scale: (M, 0) and a_scale: (M, 1)

    comp_shift: np.array, size=(M, 2)
                Bitshift (to the right) to apply for computing, b_shift: (M, 0) and a_shift: (M, 1)

    y_scale: float:
             Scale factor of the output, same as the parameter

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
        y_shift += b_shift + bit_reserve

        # modify the shift for denominators (a), they must be in the same range as the section input
        a_shift = a_shift - (n_bits - 1)

        # set the shift for the numerators (b), they get their new range
        b_shift = -bit_reserve - (n_bits - 1)

        # store the values
        scale_coeff[m, 1] = a_scale
        scale_coeff[m, 0] = b_scale
        comp_shift[m, 1] = a_shift
        comp_shift[m, 0] = b_shift

        # quantize the coefficients
        quant_coeff[m, 3:] = quantize(coeff[m, 3:], a_scale, num_bits=n_bits, do_round=True)
        quant_coeff[m, :3] = quantize(coeff[m, :3], b_scale, num_bits=n_bits, do_round=True)

    return quant_coeff, scale_coeff, comp_shift.astype(int), y_scale, y_shift


def quant_sos_filt(data, quant_filter, scale_data, mode="sosfilt", n_bits=8,
                   intermediate_bits=INTERMEDIATE_BITS, filter_bits=N_FILTER_BITS,
                   form=DEFAULT_FORM):
    """ applies sos filter in quantized form (Direct Form 1 or 2)

    Description
    -----------
    We try to fit all intermediate values in 8 bits. However, the numbers are represented as 16 bit
    values to make sure they do not overflow. Also, by doing it like this, we can still use SIMD
    operations in the Mr. Wolf implementation.

    To achieve this goal, we define different scale regions. We use the direct form 1 of the IIR
    filter (see https://en.wikipedia.org/wiki/Digital_biquad_filter#Direct_form_1). A new scale
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

    quant_filter: tuple (quant_coeff, scale_coeff, comp_shift, output_scale, output_shift):

        quant_coeff: np.array, size=(M, 6)
                     filter coefficients, with M second order sections. The first three represent
                     the numerator, and the second the denominator.

        scale_coeff: np.array, size=(M, 2)
                     Scale factors for the coefficients, where the first represents the scale factor
                     for the numerator and the second for the denominator.

        comp_shift: np.array, size=(M, 2), dtype=int
                    amount to shift during computation, for all sections, numerator and denominators

        output_scale: float
                      Scale factor of the output data

        output_shift: int
                      amount to shift the output data (to the left)

    scale_data: float
                Scale factor for the input data

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

    coeff, scale_coeff, comp_shift, y_scale, y_shift = quant_filter

    # check data
    assert len(data.shape) == 1
    assert len(coeff.shape) == 2
    assert coeff.shape[1] == 6
    assert scale_coeff.shape[0] == coeff.shape[0]
    assert scale_coeff.shape[1] == 2
    assert mode in ["valid", "same", "full", "sosfilt"]
    assert n_bits == 8
    assert form in [1, 2]

    N = data.shape[0]
    M = coeff.shape[0]

    # quantize x
    x_quant = quantize_to_int(data, scale_data, n_bits)

    # generate the coefficients
    a = np.ones((M + 1, 3), dtype=int) << (filter_bits - 1)
    b = np.ones((M + 1, 3), dtype=int) << (filter_bits - 1)
    a_shift = np.ones((M + 1, ), dtype=int) * (filter_bits - 1)
    b_shift = np.ones((M + 1, ), dtype=int) * (filter_bits - 1)

    for m in range(M):
        a[m + 1, :] = quantize_to_int(coeff[m, 3:], scale_coeff[m, 1], filter_bits, do_round=True)
        b[m + 1, :] = quantize_to_int(coeff[m, :3], scale_coeff[m, 0], filter_bits, do_round=True)
        a_shift[m + 1] = -comp_shift[m, 1]
        b_shift[m + 1] = -comp_shift[m, 0]

    # internal representation has n_bits: n_bits * 2 (because we multiply two such numbers together)
    # x = x << (n_bits - 1)

    if form == 1:
        y = quant_sos_filt_df1(x_quant, a, b, a_shift, b_shift, intermediate_bits)
    else:
        y = quant_sos_filt_df2(x_quant, a, b, a_shift, b_shift, intermediate_bits)

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
    result = dequantize_to_float(y, y_scale, n_bits)

    return result


def quant_sos_filt_df1(data, a, b, a_shift, b_shift, intermediate_bits=INTERMEDIATE_BITS):
    """ apply sos filter in direct form 1 """
    show_warning = OVERFLOW_WARNING
    M = a.shape[0]
    N = data.shape[0]
    N_prime = N + (M - 1) * 2

    # pad x such that we do not get any index errors
    x = np.zeros((N_prime, ), dtype=int)
    x[:N] = data

    clip_range = 1 << (intermediate_bits - 1)

    regs = np.zeros((M, 3), dtype=int)
    y = np.zeros((N_prime, ), dtype=int)

    # main body of the computation
    for k in range(N_prime):
        # make input registers
        # regs[0, 2] = regs[0, 1]
        # regs[0, 1] = regs[0, 0]
        regs[0, 1:] = regs[0, :-1]
        regs[0, 0] = x[k]

        # do sum of all sections
        for m in range(1, M):
            # move the registers
            # regs[m, 2] = regs[m, 1]
            # regs[m, 1] = regs[m, 0]
            regs[m, 1:] = regs[m, :-1]

            # add the input from the previous section
            acc = np.dot(regs[m - 1, :], b[m, :]) >> b_shift[m]

            # add the self reference
            acc -= np.dot(regs[m, 1:], a[m, 1:]) >> a_shift[m]

            # error handling
            if np.abs(acc) >= clip_range:
                # clip the values
                acc = np.clip(-clip_range, clip_range, acc)
                if show_warning:
                    print(f"Warning: overflow in quant_sos_filt detected!: {k=}, {m=}, {acc=}")
                    show_warning = False
                if OVERFLOW_RAISE:
                    raise RuntimeError("Overflow")

            regs[m, 0] = acc

        # store the output
        y[k] = regs[M - 1, 0]

    return y


def quant_sos_filt_df2(data, a, b, a_shift, b_shift, intermediate_bits):
    """ apply sos filter in direct form 1 """
    show_warning = OVERFLOW_WARNING
    M = a.shape[0]
    N = data.shape[0]
    N_prime = N + (M - 1) * 2

    # pad x such that we do not get any index errors
    x = np.zeros((N_prime, ), dtype=int)
    x[:N] = data

    clip_range = 1 << (intermediate_bits - 1)

    regs = np.zeros((M, 3), dtype=int)
    y = np.zeros((N_prime, ), dtype=int)

    # main body of the computation
    for k in range(N_prime):
        # get the new element
        regs[0, 0] = x[k]

        # compute all the sections
        for m in range(1, M):
            # move the registers
            regs[m, 1:] = regs[m, :-1]

            # add the input from the previous section
            acc = np.dot(regs[m - 1, :], b[m - 1, :]) >> b_shift[m - 1]

            # add the self reference
            acc -= np.dot(regs[m, 1:], a[m, 1:]) >> a_shift[m]

            # error handling
            if np.abs(acc) >= clip_range:
                # clip the values
                acc = np.clip(-clip_range, clip_range, acc)
                if show_warning:
                    print(f"Warning: overflow in quant_sos_filt detected!: {k=}, {m=}, {acc=}")
                    show_warning = False
                if OVERFLOW_RAISE:
                    raise RuntimeError("Overflow")

            regs[m, 0] = acc

        # store the output
        y[k] = np.dot(regs[M - 1, :], b[M - 1, :]) >> b_shift[M - 1]

    return y


def _plot(band, coeff):
    from scipy.signal import sosfilt
    import matplotlib.pyplot as plt

    w_below = band[0] / 2
    w_inside = (band[0] + band[1]) / 2
    w_above = band[1] * 2
    freqs = [w_below, band[0], w_inside, band[1], w_above]

    for w in freqs:
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
        plt.legend()
        plt.title(f"freq: {w}, band: {band}")
        plt.show()


def _par_measure(w, t, coeff, n_filter_bits=None, bit_reserve=None):

    if n_filter_bits is None:
        n_filter_bits = N_FILTER_BITS
    if bit_reserve is None:
        bit_reserve = BIT_RESERVE

    x = np.cos(w * t)

    y_exp_f = sosfilt(coeff, x)

    x_scale = 2
    y_scale = np.abs(y_exp_f).max()
    y_scale = 2 ** np.ceil(np.log2(y_scale))

    quant_filter = prepare_quant_filter(coeff, x_scale, y_scale, n_filter_bits, bit_reserve)

    y_acq = quant_sos_filt(x, quant_filter, x_scale, filter_bits=n_filter_bits)
    y_exp_q = sosfilt(quant_filter[0], x)

    a_acq = np.abs(y_acq[-100:]).max()
    a_exp_f = np.abs(y_exp_f[-100:]).max()
    a_exp_q = np.abs(y_exp_q[-100:]).max()

    return np.array([a_acq, a_exp_f, a_exp_q])


def _sweep(band_id, coeff, freqs=None, N=1000, T=1000, fs=250):
    if freqs is None:
        freqs = np.linspace(0, np.pi / 2, N)

    t = np.array(range(T))

    with Pool() as p:
        measure_fun = partial(_par_measure, t=t, coeff=coeff)
        measurement = list(tqdm(p.imap(measure_fun, freqs), desc=f"Band {band_id}",
                                total=len(freqs)))

    measurement = np.array(measurement)
    ampl_acq = measurement[:, 0]
    ampl_exp_f = measurement[:, 1]
    ampl_exp_q = measurement[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(freqs, ampl_exp_f, label="folat weights")
    ax.plot(freqs, ampl_exp_q, label="quantized weights")
    ax.plot(freqs, ampl_acq, label="fully quantized")
    ax.legend()
    #ax.set_yscale('log')
    plt.show()


def _find_best_params(filter_bank):
    global OVERFLOW_RAISE
    OVERFLOW_RAISE = True
    global OVERFLOW_WARNING
    OVERFLOW_WARNING = False

    filter_bits_list = [8, 9, 10, 11, 12, 13, 14, 15, 16]
    bit_reserve_list = [0]
    frequency_list = np.linspace(0, np.pi / 2, 500)
    t = np.array(range(1000))

    n_runs = len(filter_bits_list) * len(bit_reserve_list)

    scores = {}

    best_score = np.inf

    with tqdm(desc="Finding the best parameters", total=n_runs) as bar:
        with Pool() as p:

            # repeat for all parameters
            for n_filter_bits in filter_bits_list:
                for bit_reserve in bit_reserve_list:

                    # accumulated error
                    l2_error = 0
                    l1_error = 0
                    log_error = 0

                    try:
                        # repeat for all filters
                        for coeff in filter_bank:

                            # do the measurement for all frequencies
                            measure_fun = partial(_par_measure, t=t, coeff=coeff,
                                                  n_filter_bits=n_filter_bits,
                                                  bit_reserve=bit_reserve)
                            measurement = np.array(list(p.imap(measure_fun, frequency_list)))

                            exp = measurement[:, 1]
                            acq = measurement[:, 0]

                            l2_error += ((exp - acq) ** 2).sum()
                            l1_error += np.abs(exp - acq).sum()
                            log_error += np.abs(np.log(exp[5:]) - np.log(acq[5:])).sum()
                    except RuntimeError:
                        bar.update()
                        continue

                    bar.update()

                    # store the result
                    scores[(n_filter_bits, bit_reserve)] = {"l1": l1_error, "l2": l2_error, "log": log_error}
                    if l1_error < best_score:
                        best_score_tuple = (n_filter_bits, bit_reserve)
                        best_score = l1_error

    print("\nSorted after L1 error")
    for k, v in sorted(scores.items(), key=lambda t: t[1]["l1"]):
        filter_bits, bit_reserve = k
        l1 = v["l1"]
        print(f"filter bits: {filter_bits:2d}, reserve: {bit_reserve}: {l1=:.3f}")

    print("\nSorted after L2 error")
    for k, v in sorted(scores.items(), key=lambda t: t[1]["l2"]):
        filter_bits, bit_reserve = k
        l2 = v["l2"]
        print(f"filter bits: {filter_bits:2d}, reserve: {bit_reserve}: {l2=:.3f}")

    print("\nSorted after log error")
    for k, v in sorted(scores.items(), key=lambda t: t[1]["log"]):
        filter_bits, bit_reserve = k
        log = v["log"]
        print(f"filter bits: {filter_bits:2d}, reserve: {bit_reserve}: {log=:.3f}")

    best_filter_bits, best_bit_reserve = best_score_tuple
    print(f"\nBest Score: {best_filter_bits=}, {best_bit_reserve=}, {scores[best_score_tuple]}")


def _test():
    """ test function to run the frequency sweep on all filterbands """
    bands = load_bands([2], f_s=250)
    bank = load_filterbank([2], fs=250, order=2)
    # _find_best_params(bank)
    for band, coeff in zip(bands, bank):
        # _plot(band, coeff)
        _sweep(band, coeff)


if __name__ == "__main__":
    _test()
