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


def gen_stimuli(N):
    """
    This function generates the stimuli
    """
    A = np.random.randint(0, 1 << 32, (N, N))
    B = A.copy().T
    return A, B


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for N in [12, 22, 51]:

        # generate makefile
        mkf = Makefile()
        mkf.add_fc_test_source("test.c")
        mkf.add_cl_test_source("cluster.c")
        mkf.add_cl_prog_source("func/copy_mat.c")
        mkf.write()

        # generate the stimuli
        A, Y = gen_stimuli(N)

        # prepare header file
        header = HeaderFile("test_stimuli.h")
        # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
        header.add(HeaderArray("a_stm", "uint32_t", A.ravel()))
        header.add(HeaderArray("y_exp", "uint32_t", Y.ravel()))
        header.add(HeaderConstant("N_DIM", N))
        header.write()

        # compile and run
        os.system("make clean all run > {}".format(RESULT_FILE))

        # parse output
        result = parse_output(RESULT_FILE)

        casename = "N={}".format(N)

        # log the result
        logger.show_subcase_result(casename, result)

    # return summary
    return logger.summary()
