"""
This file will test the convolution implementation
"""

import random
import os
import numpy as np
import pickle
from test_utils import parse_output, TestLogger
from header_file import HeaderFile, HeaderScalar, HeaderConstant
from makefile import Makefile
import functional as F
from svd import _evd_2x2 as evd_2x2

TESTNAME = "cl::linalg::evd_2x2"
RESULT_FILE = "result.out"


def gen_stimuli(a, b, c):
    """
    This function generates the stimuli (taken from actual data)
    """
    if a is None:
        a = random.gauss(0, 1)
    a = np.float32(a)
    if b is None:
        b = random.gauss(0, 1)
    b = np.float32(b)
    if c is None:
        c = random.gauss(0, 1)
    c = np.float32(c)
    cs, sn, ev1, ev2 = evd_2x2(a, b, c)
    return a, b, c, cs, sn, ev1, ev2


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for case_a, case_b, case_c in [(None, 0, None), (0, 0, None), (None, 0, 0), (0, None, 0),
                                   (None, None, None)]:

        # generate makefile
        mkf = Makefile()
        mkf.add_fc_test_source("test.c")
        mkf.add_cl_test_source("cluster.c")
        mkf.add_cl_prog_source("linalg/svd.c")
        mkf.write()

        # generate the stimuli
        a, b, c, cs, sn, ev1, ev2 = gen_stimuli(case_a, case_b, case_c)

        # prepare header file
        header = HeaderFile("test_stimuli.h")
        # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
        header.add(HeaderScalar("a_stm", "float", a))
        header.add(HeaderScalar("b_stm", "float", b))
        header.add(HeaderScalar("c_stm", "float", c))
        header.add(HeaderScalar("cs_exp", "float", cs))
        header.add(HeaderScalar("sn_exp", "float", sn))
        header.add(HeaderScalar("ev1_exp", "float", ev1))
        header.add(HeaderScalar("ev2_exp", "float", ev2))
        header.add(HeaderConstant("EPSILON", "1.0e-35f"))
        header.write()

        # compile and run
        os.system("make clean all run > {}".format(RESULT_FILE))

        # parse output
        result = parse_output(RESULT_FILE)

        casename = "a={}, b={}, c={}".format("?" if case_a is None else case_a,
                                             "?" if case_b is None else case_b,
                                             "?" if case_c is None else case_c)

        # log the result
        logger.show_subcase_result(casename, result)

    # return summary
    return logger.summary()
