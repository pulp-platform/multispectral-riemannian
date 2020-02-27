"""
This file will test the convolution implementation
"""

import random
import os
import numpy as np
from test_utils import parse_output, TestLogger
from header_file import HeaderFile, HeaderConstant, HeaderArray, HeaderScalar
from makefile import Makefile
import functional as F

TESTNAME = "cl::func::convert_i32_to_f"
RESULT_FILE = "result.out"

MODEL_FILENAME = "../../../../data/model.pkl"
DATA_FILENAME = "../../../../data/verification.pkl"


def gen_stimuli(N):
    """
    This function generates the stimuli
    """
    scale = np.float32(random.uniform(1, 2))
    A_f = np.random.uniform(-1, 1, (N, N))
    A_f = (A_f * scale).astype(np.float32)
    A_i = F.quantize_to_int(A_f, scale, n_bits=32)
    A_q = F.dequantize(A_i, scale, n_bits=32)
    mul_factor = scale / (1 << 31)
    return A_i, A_q, mul_factor


def float_formatter(x):
    return "0x{:x}".format(F.float_as_int_repr(x))


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for N in [19, 20, 21, 22]:

        # generate makefile
        mkf = Makefile()
        mkf.add_fc_test_source("test.c")
        mkf.add_cl_test_source("cluster.c")
        mkf.add_cl_prog_source("func/convert.c")
        mkf.write()

        # generate the stimuli
        A_i, A_q, mul_factor = gen_stimuli(N)

        # prepare header file
        header = HeaderFile("test_stimuli.h")
        # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
        header.add(HeaderArray("i_stm", "int32_t", A_i.ravel()))
        header.add(HeaderArray("f_exp", "uint32_t", A_q.ravel(), formatter=float_formatter))
        header.add(HeaderConstant("N_DIM", N))
        header.add(HeaderScalar("mul_factor", "uint32_t", float_formatter(mul_factor)))
        header.add(HeaderConstant("EPSILON", logger.epsilon_str()))
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
