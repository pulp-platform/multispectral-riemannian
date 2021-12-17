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
import pickle
from test_utils import parse_output, TestLogger
from header_file import HeaderFile, HeaderConstant, HeaderArray, HeaderInclude, HeaderScalar, \
    align_array, align_array_size
from golden_model import GoldenModel
from makefile import Makefile
import functional as F

TESTNAME = "cl::mrbci::mrbci_run"
RESULT_FILE = "result.out"

MODEL_FILENAME = "../../../../data/model.pkl"
DATA_FILENAME = "../../../../data/verification.pkl"


def gen_stimuli():
    """
    This function generates the stimuli
    """
    model = GoldenModel(MODEL_FILENAME)

    with open(DATA_FILENAME, 'rb') as _f:
        data = pickle.load(_f)

    X = F.quantize_to_int(data['input_quant'], model.input_scale, model.input_n_bits, do_round=True)
    Y = model(X)

    return X, Y, model


def test(platform):
    """
    Execute the tests
    Returns: (n_total, n_success)
    """

    logger = TestLogger(TESTNAME)

    for fast_householder, parallel in [(False, False),
                                       (True, False),
                                       (True, True)]:

        # generate makefile
        mkf = Makefile(use_vega=True)
        mkf.add_fc_test_source("test.c")
        mkf.add_cl_test_source("cluster.c")
        mkf.add_cl_prog_source("mrbci/mrbci_params.c")
        mkf.add_cl_prog_source("mrbci/mrbci.c")
        mkf.add_cl_prog_source("mrbci/svm.c")
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
        X, Y, block = gen_stimuli()
        X_align = align_array(X)

        # prepare header file
        header = HeaderFile("test_stimuli.h")
        # header.add(HeaderInclude("../../../../src/cl/func/functional.h"))
        header.add(HeaderArray("x_stm", "int8_t", X_align.ravel()))
        header.add(HeaderArray("y_exp", "int32_t", Y.ravel()))
        header.write()

        # compile and run
        os.system(". $CONDA_BASE_PREFIX/etc/profile.d/conda.sh && conda deactivate && make clean all run PMSIS_OS=pulp-os platform={} > {} && conda activate mrc".format(platform, RESULT_FILE))

        # parse output
        result = parse_output(RESULT_FILE)

        casename = "naive"

        if fast_householder:
            casename = "fast hh"

        if parallel:
            casename += " + par"

        # log the result
        logger.show_subcase_result(casename, result)

    # return summary
    return logger.summary()
