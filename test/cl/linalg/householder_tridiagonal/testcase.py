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

TESTNAME = "cl::linalg::householder_tridiagonal"
RESULT_FILE = "result.out"


def gen_stimuli(N):
    """
    This function generates the stimuli (taken from actual data)
    """
    X = np.random.randn(N, 5 * N)
    A = X @ X.T
    assert A.shape == (N, N)
    assert np.all(A.T == A)

    A = A.astype(np.float32)

    L, T, R = householder_tridiagonal(A)

    assert L.dtype == np.float32
    assert T.dtype == np.float32
    assert R.dtype == np.float32

    np.testing.assert_allclose(L.T, R)

    return A, L, T, R


def float_formatter(x):
    return "0x{:x}".format(float_as_int_repr(x))


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for slow_householder, parallel in [(True, False), (False, False), (False, True)]:
        for N in [14, 16, 18, 20, 22]:

            # generate makefile
            mkf = Makefile()
            mkf.add_fc_test_source("test.c")
            mkf.add_cl_test_source("cluster.c")
            mkf.add_cl_prog_source("linalg/svd.c")
            mkf.add_cl_prog_source("linalg/svd_parallel.c")
            mkf.add_cl_prog_source("linalg/matop_f.c")
            mkf.add_cl_prog_source("func/copy_mat.c")

            if slow_householder:
                mkf.add_define("HOUSEHOLDER_SLOW")
            if parallel:
                mkf.add_define("PARALLEL")

            mkf.write()

            # generate the stimuli
            A, L, T, R = gen_stimuli(N)

            # prepare header file
            header = HeaderFile("test_stimuli.h")
            # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
            header.add(HeaderArray("a_stm", "uint32_t", A.ravel(), formatter=float_formatter))
            header.add(HeaderArray("q_exp", "uint32_t", L.ravel(), formatter=float_formatter))
            header.add(HeaderArray("t_exp", "uint32_t", T.ravel(), formatter=float_formatter))
            header.add(HeaderConstant("N_DIM", N))
            header.add(HeaderConstant("EPSILON", logger.epsilon_str()))
            header.write()

            # compile and run
            os.system(". $CONDA_BASE_PREFIX/etc/profile.d/conda.sh && conda deactivate && make clean all run > {} && conda activate mrc".format(RESULT_FILE))

            # parse output
            result = parse_output(RESULT_FILE)

            casename = "N={}".format(N)

            if not slow_householder:
                casename += " + fast HH"

            if parallel:
                casename += " + par"

            # log the result
            logger.show_subcase_result(casename, result)

    # return summary
    return logger.summary()
