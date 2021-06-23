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

TESTNAME = "cl::linalg::vecmatmul_parallel_f"
RESULT_FILE = "result.out"


def gen_stimuli(M, N):
    """
    This function generates the stimuli (taken from actual data)
    """
    a = (np.random.randn(1, M) * 100).astype(np.float32)
    B = (np.random.randn(M, N) * 100).astype(np.float32)
    y = custom_vecmatmul(a, B)
    return a, B, y


def custom_vecmatmul(a, B):
    M, N = B.shape
    y = np.zeros((1, N), dtype=np.float32)

    for n in range(N):
        acc = np.float32(0)
        for m in range(M):
            acc += a[0, m] * B[m, n]
        y[0, n] = acc
    return y


def float_formatter(x):
    return "0x{:x}".format(float_as_int_repr(x))


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for M, N in [(22, 22),
                 (21, 21),
                 (12, 12),
                 (20, 16),
                 (19, 15)]:

        # generate makefile
        mkf = Makefile()
        mkf.add_fc_test_source("test.c")
        mkf.add_cl_test_source("cluster.c")
        mkf.add_cl_prog_source("linalg/matop_f.c")
        mkf.write()

        # generate the stimuli
        a, B, y = gen_stimuli(M, N)

        # prepare header file
        header = HeaderFile("test_stimuli.h")
        # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
        header.add(HeaderArray("a_stm", "uint32_t", a.ravel(), formatter=float_formatter))
        header.add(HeaderArray("b_stm", "uint32_t", B.ravel(), formatter=float_formatter))
        header.add(HeaderArray("y_exp", "uint32_t", y.ravel(), formatter=float_formatter))
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
