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

TESTNAME = "cl::linalg::matmul_f"
RESULT_FILE = "result.out"


def gen_stimuli(M, N, O, stride_a, stride_b, stride_y):
    """
    This function generates the stimuli (taken from actual data)
    """
    A = (np.random.randn(M, stride_a) * 10).astype(np.float32)
    B = (np.random.randn(N, stride_b) * 10).astype(np.float32)
    Y_prep = np.zeros((M, stride_y), dtype=np.float32)
    Y = Y_prep.copy()
    Y[:M, :O] = A[:M, :N] @ B[:N, :O]
    return A, B, Y_prep, Y


def float_formatter(x):
    # return "{:.20e}f".format(x)
    return "0x{:x}".format(float_as_int_repr(x))


def test(platform):
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for M, N, O, stride_a, stride_b, stride_y in [(19, 20, 21, 20, 21, 21),
                                                  (20, 21, 22, 21, 22, 22),
                                                  (22, 22, 22, 22, 22, 22),
                                                  (21, 21, 21, 22, 22, 22),
                                                  (16, 16, 16, 16, 16, 22),
                                                  (22, 22, 22, 30, 22, 22),
                                                  (12, 12, 12, 22, 22, 22),
                                                  (3, 3, 3, 22, 22, 22),
                                                  (2, 2, 2, 22, 22, 22)]:

        # generate makefile
        mkf = Makefile(use_vega=True)
        mkf.add_fc_test_source("test.c")
        mkf.add_cl_test_source("cluster.c")
        mkf.add_cl_prog_source("linalg/matop_f.c")
        mkf.write()

        # generate the stimuli
        A, B, Y_prep, Y = gen_stimuli(M, N, O, stride_a, stride_b, stride_y)

        # prepare header file
        header = HeaderFile("test_stimuli.h")
        # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
        header.add(HeaderArray("a_stm", "uint32_t", A.ravel(), formatter=float_formatter))
        header.add(HeaderArray("b_stm", "uint32_t", B.ravel(), formatter=float_formatter))
        header.add(HeaderArray("y_acq", "uint32_t", Y_prep.ravel(), formatter=float_formatter))
        header.add(HeaderArray("y_exp", "uint32_t", Y.ravel(), formatter=float_formatter))
        header.add(HeaderConstant("M_DIM", M))
        header.add(HeaderConstant("N_DIM", N))
        header.add(HeaderConstant("O_DIM", O))
        header.add(HeaderConstant("STRIDE_A", stride_a))
        header.add(HeaderConstant("STRIDE_B", stride_b))
        header.add(HeaderConstant("STRIDE_Y", stride_y))
        header.add(HeaderConstant("EPSILON", logger.epsilon_str()))
        header.write()

        # compile and run
        os.system(". $CONDA_BASE_PREFIX/etc/profile.d/conda.sh && conda deactivate && make clean all run PMSIS_OS=pulp-os platform={} > {} && conda activate mrc".format(platform, RESULT_FILE))

        # parse output
        result = parse_output(RESULT_FILE)

        casename = "M={}, N={}, O={}, stride={},{},{}".format(M, N, O, stride_a, stride_b, stride_y)

        # log the result
        logger.show_subcase_result(casename, result)

    # return summary
    return logger.summary()
