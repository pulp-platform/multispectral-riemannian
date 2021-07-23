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

import random
import os
import numpy as np
import pickle
from test_utils import parse_output, TestLogger
from header_file import HeaderFile, HeaderScalar, HeaderConstant
from makefile import Makefile
import functional as F
from svd import _givens as givens_rotation

TESTNAME = "cl::linalg::givens_rotation"
RESULT_FILE = "result.out"


def gen_stimuli(a, b):
    """
    This function generates the stimuli (taken from actual data)
    """
    if a is None:
        a = random.gauss(0, 1)
    a = np.float32(a)
    if b is None:
        b = random.gauss(0, 1)
    b = np.float32(b)
    cs, sn = givens_rotation(a, b)
    return a, b, cs, sn


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for case_a, case_b in [(1, 0), (0, 1), (1, 1), (None, None)]:

        # generate makefile
        mkf = Makefile()
        mkf.add_fc_test_source("test.c")
        mkf.add_cl_test_source("cluster.c")
        mkf.add_cl_prog_source("linalg/svd.c")
        mkf.write()

        # generate the stimuli
        a, b, cs, sn = gen_stimuli(case_a, case_b)

        # prepare header file
        header = HeaderFile("test_stimuli.h")
        # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
        header.add(HeaderScalar("a_stm", "float", a))
        header.add(HeaderScalar("b_stm", "float", b))
        header.add(HeaderScalar("cs_exp", "float", cs))
        header.add(HeaderScalar("sn_exp", "float", sn))
        header.add(HeaderConstant("EPSILON", logger.epsilon_str()))
        header.write()

        # compile and run
        os.system(". ~/miniconda3/etc/profile.d/conda.sh && conda deactivate && make clean all run > {} && conda activate mrc".format(RESULT_FILE))

        # parse output
        result = parse_output(RESULT_FILE)

        casename = "a={}, b={}".format("?" if case_a is None else case_a,
                                       "?" if case_b is None else case_b)

        # log the result
        logger.show_subcase_result(casename, result)

    # return summary
    return logger.summary()
