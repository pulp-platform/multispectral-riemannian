"""
This file will test the convolution implementation
"""

import os
import numpy as np
from test_utils import parse_output, TestLogger
from header_file import HeaderFile, HeaderConstant, HeaderArray
from makefile import Makefile
from functional import float_as_int_repr

TESTNAME = "cl::linalg::matsub_f"
RESULT_FILE = "result.out"


def gen_stimuli(M, N):
    """
    This function generates the stimuli (taken from actual data)
    """
    A = (np.random.randn(M, N) * 10).astype(np.float32)
    B = (np.random.randn(M, N) * 10).astype(np.float32)
    Y = A - B
    return A, B, Y


def float_formatter(x):
    # return "{:.20e}f".format(x)
    return "0x{:x}".format(float_as_int_repr(x))


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for M, N in [(16, 20), (22, 22)]:

        # generate makefile
        mkf = Makefile()
        mkf.add_fc_test_source("test.c")
        mkf.add_cl_test_source("cluster.c")
        mkf.add_cl_prog_source("linalg/matop_f.c")
        mkf.write()

        # generate the stimuli
        A, B, Y = gen_stimuli(M, N)

        # prepare header file
        header = HeaderFile("test_stimuli.h")
        # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
        header.add(HeaderArray("a_stm", "uint32_t", A.ravel(), formatter=float_formatter))
        header.add(HeaderArray("b_stm", "uint32_t", B.ravel(), formatter=float_formatter))
        header.add(HeaderArray("y_exp", "uint32_t", Y.ravel(), formatter=float_formatter))
        header.add(HeaderConstant("M_DIM", M))
        header.add(HeaderConstant("N_DIM", N))
        header.add(HeaderConstant("EPSILON", logger.epsilon_str()))
        header.write()

        # compile and run
        os.system("make clean all run > {}".format(RESULT_FILE))

        # parse output
        result = parse_output(RESULT_FILE)

        casename = "M={}, N={}".format(M, N)

        # log the result
        logger.show_subcase_result(casename, result)

    # return summary
    return logger.summary()
