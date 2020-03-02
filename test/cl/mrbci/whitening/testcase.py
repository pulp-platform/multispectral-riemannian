"""
This file will test the convolution implementation
"""

import random
import os
import numpy as np
import pickle
from test_utils import parse_output, TestLogger
from header_file import HeaderFile, HeaderConstant, HeaderArray, HeaderInclude, HeaderScalar, \
    align_array, align_array_size
from golden_model import GoldenModel
from makefile import Makefile
import functional as F

TESTNAME = "cl::mrbci::whitening"
RESULT_FILE = "result.out"

MODEL_FILENAME = "../../../../data/model.pkl"
DATA_FILENAME = "../../../../data/verification.pkl"


def gen_stimuli(freq_idx=None):
    """
    This function generates the stimuli
    """
    model = GoldenModel(MODEL_FILENAME)

    if freq_idx is None:
        freq_idx = random.randint(0, model.n_freq - 1)

    block = model.feature_extraction.freq_band[freq_idx].whitening

    with open(DATA_FILENAME, 'rb') as _f:
        data = pickle.load(_f)

    X = F.quantize_to_int(data['cov_mat_reg_quant'][freq_idx], block.input_scale,
                          block.input_n_bits, do_round=True)
    Y = F.quantize_to_int(data['cov_mat_transform'][freq_idx], block.output_scale,
                          block.output_n_bits, do_round=True)

    return X, Y, block


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for parallel in [(False),
                     (True)]:

        if "WOLFTEST_EXHAUSTIVE" in os.environ:
            freqs_iter = list(range(18))
        else:
            freqs = list(range(18))
            random.shuffle(freqs)
            freqs_iter = freqs[:4]

        for freq_idx in freqs_iter:

            # generate makefile
            mkf = Makefile()
            mkf.add_fc_test_source("test.c")
            mkf.add_cl_test_source("cluster.c")
            mkf.add_cl_prog_source("mrbci/mrbci_params.c")
            mkf.add_cl_prog_source("mrbci/mrbci.c")
            mkf.add_cl_prog_source("mrbci/whitening.c")
            mkf.add_cl_prog_source("func/matmul.c")

            if parallel:
                mkf.add_define("PARALLEL")

            mkf.write()

            # generate the stimuli
            X, Y, _ = gen_stimuli(freq_idx)

            # prepare header file
            header = HeaderFile("test_stimuli.h")
            # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
            header.add(HeaderArray("x_stm", "int16_t", X.ravel()))
            header.add(HeaderArray("y_exp", "int32_t", Y.ravel()))
            header.add(HeaderConstant("FREQ_IDX", freq_idx))
            header.write()

            # compile and run
            os.system("make clean all run > {}".format(RESULT_FILE))

            # parse output
            result = parse_output(RESULT_FILE)

            casename = "freq {:02d}".format(freq_idx)

            if parallel:
                casename += " + par"

            # log the result
            logger.show_subcase_result(casename, result)

    # return summary
    return logger.summary()
