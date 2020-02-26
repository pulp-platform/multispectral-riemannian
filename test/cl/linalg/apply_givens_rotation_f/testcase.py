"""
This file will test the convolution implementation
"""

import os
import numpy as np
import random
from test_utils import parse_output, TestLogger
from header_file import HeaderFile, HeaderConstant, HeaderArray
from makefile import Makefile
from functional import float_as_int_repr

TESTNAME = "cl::linalg::apply_givens_rotation_f"
RESULT_FILE = "result.out"


def gen_stimuli(N, k):
    """
    This function generates the stimuli (taken from actual data)
    """

    phi = random.uniform(0, np.pi)
    cs = np.cos(phi)
    sn = np.sin(phi)

    A = np.random.randn(N, N).astype(np.float32)
    R = np.array([[cs, -sn], [sn, cs]]).astype(np.float32)

    Y = A.copy()
    Y[:, k:k + 2] = A[:, k:k + 2] @ R

    return A, cs, sn, Y


def float_formatter(x):
    # return "{:.20e}f".format(x)
    return "0x{:x}".format(float_as_int_repr(x))


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for N in [12, 22]:
        for k in [0, 2, 10]:

            # generate makefile
            mkf = Makefile()
            mkf.add_fc_test_source("test.c")
            mkf.add_cl_test_source("cluster.c")
            mkf.add_cl_prog_source("linalg/svd.c")
            mkf.write()

            # generate the stimuli
            A, cs, sn, Y = gen_stimuli(N, k)

            # prepare header file
            header = HeaderFile("test_stimuli.h")
            # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
            header.add(HeaderArray("a_stm", "uint32_t", A.ravel(), formatter=float_formatter))
            header.add(HeaderArray("rot_vec_stm", "uint32_t", [cs, sn], formatter=float_formatter))
            header.add(HeaderArray("y_exp", "uint32_t", Y.ravel(), formatter=float_formatter))
            header.add(HeaderConstant("N_DIM", N))
            header.add(HeaderConstant("K_POS", k))
            header.add(HeaderConstant("EPSILON", logger.epsilon_str()))
            header.write()

            # compile and run
            os.system("make clean all run > {}".format(RESULT_FILE))

            # parse output
            result = parse_output(RESULT_FILE)

            casename = "N={}, k={}".format(N, k)

            # log the result
            logger.show_subcase_result(casename, result)

    # return summary
    return logger.summary()
