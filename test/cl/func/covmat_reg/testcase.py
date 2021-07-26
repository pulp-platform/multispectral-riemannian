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
from header_file import HeaderFile, HeaderConstant, HeaderArray, HeaderInclude, HeaderScalar, \
    align_array, align_array_size
from golden_model import GoldenModel
from makefile import Makefile
import functional as F

TESTNAME = "cl::func::copy_transpose_mat"
RESULT_FILE = "result.out"

MODEL_FILENAME = "../../../../data/model.pkl"
DATA_FILENAME = "../../../../data/verification.pkl"


def gen_stimuli(freq_idx=None):
    """
    This function generates the stimuli
    """
    model = GoldenModel(MODEL_FILENAME)

    if freq_idx is None:
        freq_idx = random.randint(0, model.n_freq - 1)

    block = model.feature_extraction.freq_band[freq_idx].cov_mat

    with open(DATA_FILENAME, 'rb') as _f:
        data = pickle.load(_f)

    assert block.input_n_bits == 8
    assert block.output_n_bits == 16

    X = F.quantize_to_int(data['filter_out'][freq_idx], block.input_scale, block.input_n_bits,
                          do_round=True)
    Y = block(X)

    return X, Y, block


def test(platform):
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for freq_idx in [0, 1, 2, 3]:

        # generate makefile
        mkf = Makefile(use_vega=True)
        mkf.add_fc_test_source("test.c")
        mkf.add_cl_test_source("cluster.c")
        mkf.add_cl_prog_source("func/covmat.c")
        mkf.write()

        # generate the stimuli
        X, Y, block = gen_stimuli(freq_idx)
        N, M = X.shape
        assert N % 2 == 0
        X_align = align_array(X, n=4)
        M_align = align_array_size(M, n=4)

        # prepare header file
        header = HeaderFile("test_stimuli.h")
        # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
        header.add(HeaderArray("x_stm", "int8_t", X_align.ravel()))
        header.add(HeaderArray("y_exp", "int16_t", Y.ravel()))
        header.add(HeaderScalar("rho", "int32_t", block.rho))
        header.add(HeaderScalar("y_shift", "unsigned int", block.bitshift_scale))
        header.add(HeaderConstant("N_DIM", N))
        header.add(HeaderConstant("M_DIM", M))
        header.add(HeaderConstant("M_ALIGN", M_align))
        header.write()

        # compile and run
        os.system(". $CONDA_BASE_PREFIX/etc/profile.d/conda.sh && conda deactivate && make clean all run PMSIS_OS=pulp-os platform={} > {} && conda activate mrc".format(platform, RESULT_FILE))

        # parse output
        result = parse_output(RESULT_FILE)

        casename = "freq_idx: {}".format(freq_idx)

        # log the result
        logger.show_subcase_result(casename, result)

    # return summary
    return logger.summary()
