"""
This file will test the convolution implementation
"""

import random
import os
import numpy as np
from test_utils import parse_output, TestLogger
from header_file import HeaderFile, HeaderConstant, HeaderArray, HeaderScalar, align_array, \
    align_array_size
from makefile import Makefile
import functional as F

TESTNAME = "cl::func::convert_f_to_i8"
RESULT_FILE = "result.out"

MODEL_FILENAME = "../../../../data/model.pkl"
DATA_FILENAME = "../../../../data/verification.pkl"


def gen_stimuli(N):
    """
    This function generates the stimuli
    """
    scale = random.uniform(1, 2)
    A_f = np.random.uniform(-1, 1, (N, N))
    A_f = (A_f * scale).astype(np.float32)

    A_i = F.quantize_to_int(A_f, scale, n_bits=8)
    mul_factor = (1 << 7) / scale

    return A_f, align_array(A_i), mul_factor


def float_formatter(x):
    # return "0x{:x}".format(F.float_as_int_repr(x))
    return "{:.20f}".format(x)


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for N in [19, 20, 21, 22]:

        stride = align_array_size(N)

        # generate makefile
        mkf = Makefile()
        mkf.add_fc_test_source("test.c")
        mkf.add_cl_test_source("cluster.c")
        mkf.add_cl_prog_source("func/convert.c")
        mkf.write()

        # generate the stimuli
        A_f, A_i, mul_factor = gen_stimuli(N)

        # prepare header file
        header = HeaderFile("test_stimuli.h")
        # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
        header.add(HeaderArray("f_stm", "float", A_f.ravel(), formatter=float_formatter))
        header.add(HeaderArray("i_exp", "int8_t", A_i.ravel()))
        header.add(HeaderConstant("N_DIM", N))
        header.add(HeaderConstant("STRIDE", stride))
        header.add(HeaderScalar("mul_factor", "float", float_formatter(mul_factor)))
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
