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

TESTNAME = "cl::func::matmul_sqr_i16"
RESULT_FILE = "result.out"

MODEL_FILENAME = "../../../../data/model.pkl"
DATA_FILENAME = "../../../../data/verification.pkl"


def gen_stimuli(N):
    """
    This function generates the stimuli
    """
    A = np.random.randint(-(1 << 12), (1 << 12) - 1, (N, N)).astype(np.int)
    B = np.random.randint(-(1 << 12), (1 << 12) - 1, (N, N)).astype(np.int)
    Y = A @ B
    return A, B, Y


def test(platform):
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for N in [12, 16, 18, 20, 22]:

        assert N % 2 == 0

        # generate makefile
        mkf = Makefile(use_vega=True)
        mkf.add_fc_test_source("test.c")
        mkf.add_cl_test_source("cluster.c")
        mkf.add_cl_prog_source("func/matmul.c")
        mkf.write()

        # generate the stimuli
        A, B, Y = gen_stimuli(N)

        # prepare header file
        header = HeaderFile("test_stimuli.h")
        # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
        header.add(HeaderArray("a_stm", "int16_t", A.ravel()))
        header.add(HeaderArray("b_stm", "int16_t", B.ravel()))
        header.add(HeaderArray("y_exp", "int32_t", Y.ravel()))
        header.add(HeaderConstant("N_DIM", N))
        header.write()

        # compile and run
        os.system(". $CONDA_BASE_PREFIX/etc/profile.d/conda.sh && conda deactivate && make clean all run PMSIS_OS=pulp-os platform={} > {} && conda activate mrc".format(platform, RESULT_FILE))

        # parse output
        result = parse_output(RESULT_FILE)

        casename = "N={}".format(N)

        # log the result
        logger.show_subcase_result(casename, result)

    # return summary
    return logger.summary()
