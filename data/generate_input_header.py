"""
This File generates the header files defining the input data.
The following files are required
- [project_root]/data/input.npz containing the data
- [project_root]/data/config.json containing the QuantLab configuration how the network was trained
- [project_root]/data/net.npz, containing the entire network
"""

__author__ = "Tibor Schneider"
__email__ = "sctibor@student.ethz.ch"
__version__ = "0.0.2"
__date__ = "2020/01/28"
__copyright__ = """
    Copyright (C) 2020 ETH Zurich. All rights reserved.

    Author: Tibor Schneider, ETH Zurich

    SPDX-License-Identifier: Apache-2.0
    Licensed under the Apache License, Version 2.0 (the License); you may
    not use this file except in compliance with the License.
    You may obtain a copy of the License at

    www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an AS IS BASIS, WITHOUT
    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import argparse
import pickle

from header_file import HeaderFile, HeaderArray, align_array
from golden_model import GoldenModel
import functional as F

DEFAULT_MODEL_FILENAME = "model.pkl"
DEFAULT_DATA_FILENAME = "input.pkl"
DEFAULT_HEADER_NAME = "../src/cl/input.h"


def generate_input_header(model_filename: str = DEFAULT_MODEL_FILENAME,
                          input_filename: str = DEFAULT_DATA_FILENAME,
                          output_filename: str = DEFAULT_HEADER_NAME) -> None:

    # load network
    model = GoldenModel(model_filename)

    with open(input_filename, 'rb') as _f:
        data = pickle.load(_f)

    data_quant = F.quantize_to_int(data, model.input_scale, model.input_n_bits, do_round=True)
    data_quant_align = align_array(data_quant)

    # generate the header file
    header = HeaderFile(output_filename, "__INPUT_H__", with_c=True)
    header.add(HeaderArray("input_data", "int8_t", data_quant_align.ravel()))
    header.write()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Generates the header file defining the trained EEGNet")
    parser.add_argument("-o", "--output", help="Export header file name",
                        default=DEFAULT_HEADER_NAME)
    parser.add_argument("-m", "--model", help="pickle file containing the model parameters",
                        default=DEFAULT_MODEL_FILENAME)
    parser.add_argument("-i", "--input", help="numpy file containing the input",
                        default=DEFAULT_DATA_FILENAME)
    args = parser.parse_args()

    generate_input_header(args.model, args.input, args.output)
