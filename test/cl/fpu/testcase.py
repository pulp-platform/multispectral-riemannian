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

NUM_TEST = 10


def gen_stimuli():
    """
    This function generates the stimuli (taken from actual data)
    """
    a = np.random.randn(NUM_TEST).astype(np.float32)
    b = np.random.randn(NUM_TEST).astype(np.float32)
    c = np.abs(np.random.randn(NUM_TEST)).astype(np.float32)
    d = np.random.randint(-(1 << 31), (1 << 31) - 1, (NUM_TEST, )).astype(int)
    return a, b, c, d


def float_formatter(x):
    # return "{:.20e}f".format(x)
    return "0x{:x}".format(float_as_int_repr(x))


def compare_result(result, case_name, exp, rel_tol=0):
    max_rel_diff = 0
    mean_rel_diff = 0
    for i in range(NUM_TEST):
        acq_str = result[case_name][str(i)]
        acq = float_from_int_repr(int(acq_str, 16))
        exp_d = np.float64(exp[i])
        acq_d = np.float64(acq)
        abs_diff = np.abs(acq_d - exp_d)
        rel_diff = abs_diff / np.abs(exp_d)
        max_rel_diff = max(max_rel_diff, rel_diff)
        mean_rel_diff += rel_diff / NUM_TEST
        del result[case_name][str(i)]

    success = max_rel_diff <= rel_tol
    result[case_name]["mean_rel_diff"] = "{:.2e}".format(mean_rel_diff)
    result[case_name]["max_rel_diff"] = "{:.2e}".format(max_rel_diff)
    result[case_name]["result"] = success


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
    mkf.write()

    # generate the stimuli
    a, b, c, d = gen_stimuli()

    # prepare header file
    header = HeaderFile("test_stimuli.h")
    # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
    header.add(HeaderArray("a_stm", "uint32_t", a.ravel(), formatter=float_formatter))
    header.add(HeaderArray("b_stm", "uint32_t", b.ravel(), formatter=float_formatter))
    header.add(HeaderArray("c_stm", "uint32_t", c.ravel(), formatter=float_formatter))
    header.add(HeaderArray("d_stm", "int32_t", d.ravel()))
    header.add(HeaderConstant("LENGTH", NUM_TEST))
    header.write()

    # compile and run
    os.system("make clean all run > {}".format(RESULT_FILE))

    # parse output
    result = parse_output(RESULT_FILE)

    # read the results out
    compare_result(result, "add", a + b, rel_tol=logger.epsilon)
    compare_result(result, "sub", a - b, rel_tol=logger.epsilon)
    compare_result(result, "mul", a * b, rel_tol=logger.epsilon)
    compare_result(result, "div", a / b, rel_tol=logger.epsilon)
    compare_result(result, "sqrt", np.sqrt(c), rel_tol=logger.epsilon)
    compare_result(result, "madd", (a * b) + c, rel_tol=logger.epsilon)
    compare_result(result, "msub", (a * b) - c, rel_tol=logger.epsilon)
    compare_result(result, "nmadd", -((a * b) + c), rel_tol=logger.epsilon)
    compare_result(result, "nmsub", -((a * b) - c), rel_tol=logger.epsilon)
    compare_result(result, "fcvt", d.astype(np.float32), rel_tol=logger.epsilon)

    subcase_name = "FPU"

    # log the result
    logger.show_subcase_result(subcase_name, result)

    # return summary
    return logger.summary()
