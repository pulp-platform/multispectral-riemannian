"""
This file will test the convolution implementation
"""

import random
import os
import numpy as np
import pickle
from test_utils import parse_output, TestLogger
from header_file import HeaderFile, HeaderConstant, HeaderArray, HeaderInclude
from golden_model import GoldenModel
from makefile import Makefile
import functional as F

TESTNAME = "cl::func::swap_mat"
RESULT_FILE = "result.out"

MODEL_FILENAME = "../../../../data/model.pkl"
DATA_FILENAME = "../../../../data/verification.pkl"


def gen_stimuli(N, M, stride_a, stride_b):
    """
    This function generates the stimuli
    """
    A = np.random.randint(0, 1 << 32, (N, stride_a))
    B = np.random.randint(0, 1 << 32, (N, stride_b))
    BS = B.copy()
    BS[:N, :M] = A[:N, :M]
    return A, B, BS


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for N, M, stride_a, stride_b in [(22, 12, 22, 12),
                                     (12, 22, 30, 22),
                                     (22, 22, 22, 30),
                                     (22, 22, 30, 40),
                                     (22, 21, 35, 31)]:

        # generate makefile
        mkf = Makefile()
        mkf.add_fc_test_source("test.c")
        mkf.add_cl_test_source("cluster.c")
        mkf.add_cl_prog_source("func/copy_mat.c")
        mkf.write()

        # generate the stimuli
        A, B, BS = gen_stimuli(N, M, stride_a, stride_b)

        # prepare header file
        header = HeaderFile("test_stimuli.h")
        # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
        header.add(HeaderArray("a_vec", "uint32_t", A.ravel()))
        header.add(HeaderArray("b_vec", "uint32_t", B.ravel()))
        header.add(HeaderArray("b_exp_vec", "uint32_t", BS.ravel()))
        header.add(HeaderConstant("N_DIM", N))
        header.add(HeaderConstant("M_DIM", M))
        header.add(HeaderConstant("STRIDE_A", stride_a))
        header.add(HeaderConstant("STRIDE_B", stride_b))
        header.write()

        # compile and run
        os.system("make clean all run > {}".format(RESULT_FILE))

        # parse output
        result = parse_output(RESULT_FILE)

        casename = "N={}, M={}, stride_a={}, stride_b={}".format(N, M, stride_a, stride_b)

        # log the result
        logger.show_subcase_result(casename, result)

    # return summary
    return logger.summary()