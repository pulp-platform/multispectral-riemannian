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

TESTNAME = "cl::linalg::matmul_f (submat)"
RESULT_FILE = "result.out"


def gen_stimuli(submat_size, mat_size, right_special):
    """
    This function generates the stimuli (taken from actual data)
    """
    A = (np.random.randn(mat_size, mat_size) * 10).astype(np.float32)
    B = np.eye(mat_size).astype(np.float32)
    k = mat_size - submat_size
    B[k:, k:] = (np.random.randn(submat_size, submat_size) * 10).astype(np.float32)

    if not right_special:
        t = A
        A = B
        B = t

    Y = A @ B
    return A, B, Y


def float_formatter(x):
    # return "{:.20e}f".format(x)
    return "0x{:x}".format(float_as_int_repr(x))


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for right_special in [True, False]:
        for submat_size, mat_size in [(12, 22), (18, 22), (21, 22)]:

            # generate makefile
            mkf = Makefile()
            mkf.add_fc_test_source("test.c")
            mkf.add_cl_test_source("cluster.c")
            mkf.add_cl_prog_source("linalg/matop_f.c")
            mkf.add_cl_prog_source("func/copy_mat.c")
            mkf.write()

            # generate the stimuli
            A, B, Y = gen_stimuli(submat_size, mat_size, right_special)

            # prepare header file
            header = HeaderFile("test_stimuli.h")
            # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
            header.add(HeaderArray("a_stm", "uint32_t", A.ravel(), formatter=float_formatter))
            header.add(HeaderArray("b_stm", "uint32_t", B.ravel(), formatter=float_formatter))
            header.add(HeaderArray("y_exp", "uint32_t", Y.ravel(), formatter=float_formatter))
            header.add(HeaderConstant("MAT_DIM", mat_size))
            header.add(HeaderConstant("SUBMAT_DIM", submat_size))
            header.add(HeaderConstant("RIGHT_SPECIAL", 1 if right_special else 0))
            header.add(HeaderConstant("EPSILON", logger.epsilon_str()))
            header.write()

            # compile and run
            os.system(". $CONDA_BASE_PREFIX/etc/profile.d/conda.sh && conda deactivate && make clean all run > {} && conda activate mrc".format(RESULT_FILE))

            # parse output
            result = parse_output(RESULT_FILE)

            casename = "N={}, submat: {}, {}".format(mat_size, submat_size, "right" if right_special else "left")

            # log the result
            logger.show_subcase_result(casename, result)

    # return summary
    return logger.summary()
