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
import numpy as np
import os
import pickle
from test_utils import parse_output, TestLogger
from header_file import HeaderFile, HeaderConstant, HeaderArray, HeaderInclude, HeaderScalar, \
    align_array, align_array_size
from golden_model import GoldenModel
from makefile import Makefile
import functional as F

TESTNAME = "cl::mrbci::extract_features_in_band"
RESULT_FILE = "result.out"

MODEL_FILENAME = "../../../../data/model.pkl"
DATA_FILENAME = "../../../../data/verification.pkl"

def gen_stimuli(freq_idx):
    """
    This function generates the stimuli
    """
    model = GoldenModel(MODEL_FILENAME)
    block = model.feature_extraction.freq_band[freq_idx]

    with open(DATA_FILENAME, 'rb') as _f:
        data = pickle.load(_f)

    X = F.quantize_to_int(data['input_quant'], block.input_scale, block.input_n_bits, do_round=True)
    Y = block(X)

    return X, Y


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for fast_householder, parallel in [(False, False),
                                       (True, False),
                                       (True, True)]:

        if "WOLFTEST_EXHAUSTIVE" in os.environ:
            freqs_iter = list(range(18))
        else:
            freqs = list(range(18))
            random.shuffle(freqs)
            freqs_iter = freqs[:4]

        for freq_idx in freqs_iter:

            # generate makefile
            mkf = Makefile()
            mkf.add_fc_test_source("test.c")
            mkf.add_cl_test_source("cluster.c")
            mkf.add_cl_prog_source("mrbci/mrbci_params.c")
            mkf.add_cl_prog_source("mrbci/mrbci.c")
            mkf.add_cl_prog_source("mrbci/filter.c")
            mkf.add_cl_prog_source("mrbci/covmat.c")
            mkf.add_cl_prog_source("mrbci/whitening.c")
            mkf.add_cl_prog_source("mrbci/logm.c")
            mkf.add_cl_prog_source("mrbci/half_diag.c")
            mkf.add_cl_prog_source("mrbci/feature_extraction.c")
            mkf.add_cl_prog_source("func/sos_filt.c")
            mkf.add_cl_prog_source("func/copy_mat.c")
            mkf.add_cl_prog_source("func/covmat.c")
            mkf.add_cl_prog_source("func/matmul.c")
            mkf.add_cl_prog_source("func/convert.c")
            mkf.add_cl_prog_source("linalg/matop_f.c")
            mkf.add_cl_prog_source("linalg/svd.c")
            mkf.add_cl_prog_source("linalg/svd_parallel.c")

            if not fast_householder:
                mkf.add_define("HOUSEHOLDER_SLOW")

            if parallel:
                mkf.add_define("PARALLEL")

            mkf.write()

            # generate the stimuli
            # X, Y = gen_stimuli(freq_idx)
            X, Y = gen_stimuli(freq_idx)
            X_align = align_array(X)

            # prepare header file
            header = HeaderFile("test_stimuli.h")
            # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
            header.add(HeaderArray("x_stm", "int8_t", X_align.ravel()))
            header.add(HeaderArray("y_exp", "int8_t", Y.ravel()))
            header.add(HeaderConstant("FREQ_IDX", freq_idx))
            header.write()

            # compile and run
            os.system(". $CONDA_BASE_PREFIX/etc/profile.d/conda.sh && conda deactivate && make clean all run > {} && conda activate mrc".format(RESULT_FILE))

            # parse output
            result = parse_output(RESULT_FILE)

            casename = "freq {:02d}".format(freq_idx)

            if fast_householder:
                casename += " + fast hh"

            if parallel:
                casename += " + par"

            # log the result
            logger.show_subcase_result(casename, result)

    # return summary
    return logger.summary()
