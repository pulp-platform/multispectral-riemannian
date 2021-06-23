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
from svd import _vec_norm as vec_norm

TESTNAME = "cl::linalg::vnorm_f"
RESULT_FILE = "result.out"


def gen_stimuli(N, stride=1):
    """
    This function generates the stimuli (taken from actual data)
    """
    a = (np.random.randn(N, stride) * 10).astype(np.float32)
    norm = vec_norm(a[:, 0])
    return a, norm


def float_formatter(x):
    return "0x{:x}".format(float_as_int_repr(x))


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for N, stride in [(8, 22), (22, 1), (22, 4)]:

        # generate makefile
        mkf = Makefile()

        mkf.add_fc_test_source("test.c")
        mkf.add_cl_test_source("cluster.c")
        mkf.add_cl_prog_source("linalg/matop_f.c")

        mkf.write()

        # generate the stimuli
        a, norm = gen_stimuli(N, stride)

        # prepare header file
        header = HeaderFile("test_stimuli.h")
        # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
        header.add(HeaderArray("a_stm", "uint32_t", a.ravel(), formatter=float_formatter))
        header.add(HeaderArray("y_exp", "uint32_t", [norm], formatter=float_formatter))
        header.add(HeaderConstant("DIM", N))
        header.add(HeaderConstant("STRIDE", stride))
        header.add(HeaderConstant("EPSILON", logger.epsilon_str()))
        header.write()

        # compile and run
        os.system("make clean all run > {}".format(RESULT_FILE))

        # parse output
        result = parse_output(RESULT_FILE)

        casename = "N={}, stride={}".format(N, stride)

        # log the result
        logger.show_subcase_result(casename, result)

    # return summary
    return logger.summary()
