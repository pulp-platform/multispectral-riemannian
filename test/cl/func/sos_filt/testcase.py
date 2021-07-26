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
from header_file import HeaderFile, HeaderConstant, HeaderArray, HeaderInclude
from golden_model import GoldenModel
from makefile import Makefile
import functional as F

TESTNAME = "cl::func::sos_filt"
RESULT_FILE = "result.out"

MODEL_FILENAME = "../../../../data/model.pkl"
DATA_FILENAME = "../../../../data/verification.pkl"


def gen_stimuli(size=875, ch=None, freq_idx=None):
    """
    This function generates the stimuli (taken from actual data)
    """
    model = GoldenModel(MODEL_FILENAME)
    with open(DATA_FILENAME, 'rb') as _f:
        data = pickle.load(_f)

    # get frequency id and channel
    if freq_idx is None:
        freq_idx = random.randrange(model.n_freq)
    if ch is None:
        ch = random.randrange(22)

    filt = model.feature_extraction.freq_band[freq_idx].filter
    x_vec = data['input_quant']
    x_vec = F.quantize_to_int(x_vec, filt.input_scale, filt.input_n_bits)
    y_vec = filt(x_vec)
    return x_vec[ch], y_vec[ch], filt


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for size in [100, 512, 875]:

        # generate makefile
        mkf = Makefile()
        mkf.add_fc_test_source("test.c")
        mkf.add_cl_test_source("cluster.c")
        mkf.add_cl_prog_source("func/sos_filt.c")
        mkf.write()

        # generate the stimuli
        x_vec, exp_vec, filt = gen_stimuli(size)

        # prepare header file
        header = HeaderFile("test_stimuli.h")
        # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
        header.add(HeaderConstant("LENGTH", size))
        header.add(HeaderArray("x_vec", "int8_t", x_vec))
        header.add(HeaderArray("exp_vec", "int8_t", exp_vec))
        filt.add_header_entries(header, "test_filt", is_full_name=True)
        header.write()

        # compile and run
        os.system(". $CONDA_BASE_PREFIX/etc/profile.d/conda.sh && conda deactivate && make clean all run > {} && conda activate mrc".format(RESULT_FILE))

        # parse output
        result = parse_output(RESULT_FILE)

        casename = "size={}".format(size)

        # log the result
        logger.show_subcase_result(casename, result)

    # return summary
    return logger.summary()
