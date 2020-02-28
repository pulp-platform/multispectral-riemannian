"""
This File generates the header files defining the trained and quantized network.
The following files are required
- [project_root]/data/config.json containing the QuantLab configuration how the network was trained
- [project_root]/data/net.npz, containing the entire network
"""

__author__ = "Tibor Schneider"
__email__ = "sctibor@student.ethz.ch"
__version__ = "0.0.1"
__date__ = "2020/02/28"

import argparse
import numpy as np
from header_file import HeaderFile, HeaderConstant, HeaderArray, HeaderComment, HeaderScalar, \
    HeaderInclude, HeaderStruct, align_array, align_array_size
from golden_model import GoldenModel
import functional as F

DEFAULT_MODEL_FILENAME = "model.pkl"
DEFAULT_DATA_FILENAME = "verification.pkl"
DEFAULT_HEADER_NAME = "../src/cl/mrbci/mrbci_params.h"


def float_formatter(x):
    return "0x{:x}".format(F.float_as_int_repr(x))


def generate_mrbci_header(model_filename: str = DEFAULT_MODEL_FILENAME,
                          output_filename: str = DEFAULT_HEADER_NAME) -> None:

    # load the golden model
    model = GoldenModel(model_filename)
    # first frequency band
    first_band = model.feature_extraction.freq_band[0]

    # check dimensionality
    C, T = model.input_shape
    assert C % 2 == 0

    # prepare the header file
    header_file = HeaderFile(output_filename, with_c=True, define_guard="__CL_MRBCI_PARAMS_H__")
    header_file.add(HeaderInclude("../func/functional.h"))
    header_file.add(HeaderInclude("../linalg/linalg.h"))
    header_file.add(HeaderComment("Multiscale Riemannian Brain Computer Interface", mode="/*"))

    # add dimensions
    header_file.add(HeaderComment("Dimensionality", blank_line=True))
    header_file.add(HeaderConstant("MRBCI_T", T, blank_line=False))
    header_file.add(HeaderConstant("MRBCI_T_ALIGN", align_array_size(T), blank_line=False))
    header_file.add(HeaderConstant("MRBCI_C", C, blank_line=False))
    header_file.add(HeaderConstant("MRBCI_C_ALIGN", align_array_size(C), blank_line=False))
    header_file.add(HeaderConstant("MRBCI_NUM_FREQ", model.n_freq, blank_line=False))
    header_file.add(HeaderConstant("MRBCI_NUM_CLASS", model.output_shape[0], blank_line=True))

    # filters
    header_file.add(HeaderComment("Filters", blank_line=True, mode="/*"))
    # generate all structs
    structs = [band.filter.get_initializer_str(double_tab=True)
               for band in model.feature_extraction.freq_band]
    header_file.add(HeaderArray("mrbci_filter_params", "func_sos_filt_2S_params_t", structs,
                                skip_format=True))

    # CovMat
    header_file.add(HeaderComment("CovMat", mode="/*", blank_line=True))
    rho_list = [band.cov_mat.rho for band in model.feature_extraction.freq_band]
    shift_list = [band.cov_mat.bitshift_scale for band in model.feature_extraction.freq_band]
    header_file.add(HeaderArray("mrbci_covmat_rho", "int32_t", rho_list, blank_line=True))
    header_file.add(HeaderArray("mrbci_covmat_shift", "unsigned int", shift_list, blank_line=True))

    # Whitening
    header_file.add(HeaderComment("Whitening", mode="/*", blank_line=True))
    c_ref_array = np.array([band.whitening.ref_invsqrtm
                            for band in model.feature_extraction.freq_band])
    assert c_ref_array.shape == (model.n_freq, C, C)
    header_file.add(HeaderArray("mrbci_c_ref_invsqrtm_i16", "int16_t", c_ref_array.ravel()))
    header_file.add(HeaderArray("mrbci_c_ref_invsqrtm_i32", "int32_t", c_ref_array.ravel()))

    # Logm
    header_file.add(HeaderComment("Logm", mode="/*", blank_line=True))
    dequant_array = [np.float32(band.logm.input_scale / (1 << 31))
                     for band in model.feature_extraction.freq_band]
    requant_factor = np.float32((1 << 7) / first_band.logm.output_scale)
    header_file.add(HeaderArray("mrbci_logm_dequant_i", "uint32_t", dequant_array,
                                formatter=float_formatter, blank_line=False))
    header_file.add(HeaderScalar("mrbci_logm_requant_i", "uint32_t",
                                 float_formatter(requant_factor), blank_line=True))

    # half diagonalization
    header_file.add(HeaderComment("Half Diagonalization", mode="/*", blank_line=True))
    header_file.add(HeaderConstant("MRBCI_HALF_DIAG_FEATURES",
                                   first_band.half_diag.output_shape[0]))
    header_file.add(HeaderConstant("MRBCI_HALF_DIAG_SQRT2", first_band.half_diag.sqrt2))
    header_file.add(HeaderConstant("MRBCI_HALF_DIAG_SHIFT", first_band.half_diag.bitshift_scale))
    header_file.add(HeaderConstant("MRBCI_HALF_DIAG_SHIFT_DIAG",
                                   first_band.half_diag.bitshift_scale_diag))

    # SVM
    header_file.add(HeaderComment("Support Vector Machine", mode="/*", blank_line=True))
    header_file.add(HeaderConstant("MRBCI_SVM_NUM_FEATURES", model.svm.input_shape[0]))
    header_file.add(HeaderConstant("MRBCI_SVM_NUM_FEATURES_ALIGN",
                                   align_array_size(model.svm.input_shape[0])))
    header_file.add(HeaderArray("mrbci_svm_weights", "int8_t",
                                align_array(model.svm.weight).ravel()))
    header_file.add(HeaderArray("mrbci_svm_bias", "int32_t", model.svm.bias.ravel()))

    # write header
    header_file.write()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generates the header file for the Multiscale Riemannian BCI")
    parser.add_argument("-o", "--output", help="Export header file name",
                        default=DEFAULT_HEADER_NAME)
    parser.add_argument("-m", "--model", help="pickle file containing the model parameters",
                        default=DEFAULT_MODEL_FILENAME)
    args = parser.parse_args()

    generate_mrbci_header(args.model, args.output)
