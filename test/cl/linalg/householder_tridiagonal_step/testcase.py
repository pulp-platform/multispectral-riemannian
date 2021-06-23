"""
This file will test the convolution implementation
"""

"""
 * Copyright (C) 2020 ETH Zurich. All rights reserved.
 *
 * Author: Tibor Schneider, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

import os
import numpy as np
from test_utils import parse_output, TestLogger
from header_file import HeaderFile, HeaderArray, HeaderConstant
from makefile import Makefile
from svd import _householder_tridiagonal as householder_tridiagonal
from functional import float_as_int_repr

TESTNAME = "cl::linalg::householder_tridiagonal_step"
RESULT_FILE = "result.out"


def gen_stimuli(N, kp1):
    """
    This function generates the stimuli (taken from actual data)
    """
    X = np.random.randn(N, 5 * N)
    A = X @ X.T
    A = A.astype(np.float32)

    # generate a random v and w
    v = np.random.randn(N, 1).astype(np.float32)
    w = np.random.randn(N, 1).astype(np.float32)

    v_zeros = v.copy()
    v_zeros[:kp1, 0] = np.float32(0)

    c = (v @ w.T)[0, 0]

    Y = A - 2 * (v_zeros @ w.T + w @ v_zeros.T) + 4 * c * v_zeros @ v_zeros.T

    return A, v, w, c, Y


def float_formatter(x):
    return "0x{:x}".format(float_as_int_repr(x))


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for N, kp1 in [(22, 1), (22, 5), (22, 8)]:

        # generate makefile
        mkf = Makefile()
        mkf.add_fc_test_source("test.c")
        mkf.add_cl_test_source("cluster.c")
        mkf.add_cl_prog_source("linalg/svd.c")
        mkf.add_cl_prog_source("linalg/matop_f.c")
        mkf.write()

        # generate the stimuli
        A, v, w, c, Y = gen_stimuli(N, kp1)

        # prepare header file
        header = HeaderFile("test_stimuli.h")
        # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
        header.add(HeaderArray("a_stm", "uint32_t", A.ravel(), formatter=float_formatter))
        header.add(HeaderArray("v_stm", "uint32_t", v.ravel(), formatter=float_formatter))
        header.add(HeaderArray("w_stm", "uint32_t", w.ravel(), formatter=float_formatter))
        header.add(HeaderArray("c_stm", "uint32_t", [c], formatter=float_formatter))
        header.add(HeaderArray("y_exp", "uint32_t", Y.ravel(), formatter=float_formatter))
        header.add(HeaderConstant("N_DIM", N))
        header.add(HeaderConstant("KP1", kp1))
        header.add(HeaderConstant("EPSILON", "1.e-5"))
        header.write()

        # compile and run
        os.system("make clean all run > {}".format(RESULT_FILE))

        # parse output
        result = parse_output(RESULT_FILE)

        casename = "N={}, k+1={}".format(N, kp1)

        # log the result
        logger.show_subcase_result(casename, result)

    # return summary
    return logger.summary()
