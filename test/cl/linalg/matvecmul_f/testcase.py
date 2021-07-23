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

TESTNAME = "cl::linalg::matvecmul_f"
RESULT_FILE = "result.out"


def gen_stimuli(M, N, stride_a):
    """
    This function generates the stimuli (taken from actual data)
    """
    A = (np.random.randn(M, stride_a) * 100).astype(np.float32)
    b = (np.random.randn(N, 1) * 100).astype(np.float32)
    y = custom_matvecmul(A, b)
    return A, b, y


def custom_matvecmul(A, b):
    M = A.shape[0]
    N = b.shape[0]
    y = np.zeros((M, 1), dtype=np.float32)

    for m in range(M):
        acc = np.float32(0)
        for n in range(N):
            acc += A[m, n] * b[n, 0]
        y[m, 0] = acc
    return y


def float_formatter(x):
    return "0x{:x}".format(float_as_int_repr(x))


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for M, N, stride_a in [(22, 22, 22),
                           (21, 21, 21),
                           (12, 12, 12),
                           (22, 16, 22),
                           (16, 16, 22)]:

        # generate makefile
        mkf = Makefile()
        mkf.add_fc_test_source("test.c")
        mkf.add_cl_test_source("cluster.c")
        mkf.add_cl_prog_source("linalg/matop_f.c")
        mkf.write()

        # generate the stimuli
        A, b, y = gen_stimuli(M, N, stride_a)

        # prepare header file
        header = HeaderFile("test_stimuli.h")
        # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
        header.add(HeaderArray("a_stm", "uint32_t", A.ravel(), formatter=float_formatter))
        header.add(HeaderArray("b_stm", "uint32_t", b.ravel(), formatter=float_formatter))
        header.add(HeaderArray("y_exp", "uint32_t", y.ravel(), formatter=float_formatter))
        header.add(HeaderConstant("M_DIM", M))
        header.add(HeaderConstant("N_DIM", N))
        header.add(HeaderConstant("STRIDE_A", stride_a))
        header.add(HeaderConstant("EPSILON", logger.epsilon_str()))
        header.write()

        # compile and run
        os.system(". ~/miniconda3/etc/profile.d/conda.sh && conda deactivate && make clean all run > {} && conda activate mrc".format(RESULT_FILE))

        # parse output
        result = parse_output(RESULT_FILE)

        casename = "M={}, N={}, stride={}".format(M, N, stride_a)

        # log the result
        logger.show_subcase_result(casename, result)

    # return summary
    return logger.summary()
