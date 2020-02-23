"""
This file will test the convolution implementation
"""

import os
import numpy as np
from test_utils import parse_output, TestLogger
from header_file import HeaderFile, HeaderConstant, HeaderArray
from makefile import Makefile
from functional import float_as_int_repr, float_from_int_repr

TESTNAME = "FPU of Mr Wolf"
RESULT_FILE = "result.out"


def gen_stimuli():
    """
    This function generates the stimuli (taken from actual data)
    """
    x = np.random.randn(9).astype(np.float32)
    x[8] = np.abs(x[8])
    y = np.array([x[0] + x[1],
                  x[2] - x[3],
                  x[4] * x[5],
                  x[6] / x[7],
                  np.sqrt(x[8])])
    return x, y


def float_formatter(x):
    # return "{:.20e}f".format(x)
    return "0x{:x}".format(float_as_int_repr(x))


def compare_result(result, case_name, exp, rel_tol=0):
    acq_str = result[case_name]["res"]
    acq = float_from_int_repr(int(acq_str, 16))
    exp_d = np.float64(exp)
    acq_d = np.float64(acq)
    abs_diff = np.abs(acq_d - exp_d)
    rel_diff = abs_diff / np.abs(exp_d)
    success = rel_diff <= rel_tol
    result[case_name]["abs_diff"] = "{:.2e}".format(abs_diff)
    result[case_name]["rel_diff"] = "{:.2e}".format(rel_diff)
    result[case_name]["result"] = success
    del result[case_name]["res"]


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    # generate makefile
    mkf = Makefile()
    mkf.add_fc_test_source("test.c")
    mkf.add_cl_test_source("cluster.c")
    mkf.add_cl_prog_source("linalg/matmul.c")
    mkf.write()

    # generate the stimuli
    x, y_exp = gen_stimuli()

    # prepare header file
    header = HeaderFile("test_stimuli.h")
    # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
    header.add(HeaderArray("stm", "uint32_t", x.ravel(), formatter=float_formatter))
    header.write()

    # compile and run
    os.system("make clean all run > {}".format(RESULT_FILE))

    # parse output
    result = parse_output(RESULT_FILE)

    # read the results out
    compare_result(result, "add", y_exp[0])
    compare_result(result, "sub", y_exp[1])
    compare_result(result, "mul", y_exp[2])
    compare_result(result, "div", y_exp[3])
    compare_result(result, "sqrt", y_exp[4])

    # log the result
    logger.show_subcase_result("FPU", result)

    # return summary
    return logger.summary()
