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

TESTNAME = "cl::linalg::vec_innerprod_f"
RESULT_FILE = "result.out"


def gen_stimuli(N):
    """
    This function generates the stimuli (taken from actual data)
    """
    a = (np.random.randn(1, N) * 100).astype(np.float32)
    b = (np.random.randn(N, 1) * 100).astype(np.float32)

    y = np.float32(0)
    for n in range(N):
        y += a[0, n] * b[n, 0]

    return a, b, y


def float_formatter(x):
    return "0x{:x}".format(float_as_int_repr(x))


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for N in [12, 13, 21, 22]:

        # generate makefile
        mkf = Makefile()
        mkf.add_fc_test_source("test.c")
        mkf.add_cl_test_source("cluster.c")
        mkf.add_cl_prog_source("linalg/matop_f.c")
        mkf.write()

        # generate the stimuli
        a, b, y = gen_stimuli(N)

        # prepare header file
        header = HeaderFile("test_stimuli.h")
        # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
        header.add(HeaderArray("a_stm", "uint32_t", a.ravel(), formatter=float_formatter))
        header.add(HeaderArray("b_stm", "uint32_t", b.ravel(), formatter=float_formatter))
        header.add(HeaderArray("y_exp", "uint32_t", [y], formatter=float_formatter))
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
