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
from header_file import HeaderFile, HeaderConstant, HeaderArray
from makefile import Makefile
from functional import float_as_int_repr

TESTNAME = "cl::linalg::matmul_to_sym_f"
RESULT_FILE = "result.out"


def gen_stimuli(N):
    """
    This function generates the stimuli (taken from actual data)
    """
    X = np.random.randn(N, 5 * N)
    T = X @ X.T

    # do a partial step of the householder tridiagonalization
    k = 0
    s = np.linalg.norm(T[k + 1:, k])
    val = T[k + 1, k]
    sign = np.sign(val)
    z = (1.0 + sign * val / s) / 2.0
    sqrtz = np.sqrt(z)
    v = np.zeros(N)
    v[k + 1] = sqrtz
    v[k + 2:] = (sign * T[k, k + 2]) / (2.0 * s * sqrtz)
    v = v.reshape(-1, 1)
    H = np.eye(N) - 2.0 * v @ v.T

    A = (H @ T).astype(np.float32)
    B = H.astype(np.float32)
    Y = A @ B

    # make diagonal
    for i in range(1, N):
        for j in range(i):
            Y[j, i] = Y[i, j]

    return A, B, Y


def float_formatter(x):
    return "0x{:x}".format(float_as_int_repr(x))


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for N in [20, 21, 22, 23]:

        # generate makefile
        mkf = Makefile()
        mkf.add_fc_test_source("test.c")
        mkf.add_cl_test_source("cluster.c")
        mkf.add_cl_prog_source("linalg/matop_f.c")
        mkf.write()

        # generate the stimuli
        A, B, Y = gen_stimuli(N)

        # prepare header file
        header = HeaderFile("test_stimuli.h")
        # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
        header.add(HeaderArray("a_stm", "uint32_t", A.ravel(), formatter=float_formatter))
        header.add(HeaderArray("b_stm", "uint32_t", B.ravel(), formatter=float_formatter))
        header.add(HeaderArray("y_exp", "uint32_t", Y.ravel(), formatter=float_formatter))
        header.add(HeaderConstant("N_DIM", N))
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
