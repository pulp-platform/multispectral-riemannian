"""
Test the golden model
"""

import numpy as np
import pickle
from collections import OrderedDict
from test_utils import TestLogger
from golden_model import GoldenModel, SVM, FeatureExtraction, RiemannianFeature, Filter, CovMat, \
    Whitening, Logm, HalfDiag
import functional as F

# TODO add a meaningful name for the test
TESTNAME = "python::GoldenModel"
MODEL_FILENAME = "../../../data/model.pkl"
DATA_FILENAME = "../../../data/verification.pkl"

TOLERANCE = 6


def test():
    """
    Execute the tests
    Returns: (n_total, n_success)
    """
    logger = TestLogger(TESTNAME)

    with open(MODEL_FILENAME, 'rb') as _f:
        model_dict = pickle.load(_f)
    with open(DATA_FILENAME, 'rb') as _f:
        data = pickle.load(_f)

    result = OrderedDict()
    result.update(test_riemannian_block(model_dict, data, Filter,
                                        'input_quant', 'filter_out'))
    result.update(test_riemannian_block(model_dict, data, CovMat,
                                        'filter_out', 'cov_mat_reg'))
    result.update(test_riemannian_block(model_dict, data, Whitening,
                                        'cov_mat_reg', 'cov_mat_transform'))
    result.update(test_riemannian_block(model_dict, data, Logm,
                                        'cov_mat_transform', 'cov_mat_logm'))
    result.update(test_riemannian_block(model_dict, data, HalfDiag,
                                        'cov_mat_logm', 'features'))
    logger.show_subcase_result('Individual Blocks:', result)

    # return summary
    return logger.summary()


def test_riemannian_block(model_dict, data, block_class, input_name, output_name):
    '''
    Test a block inside the riemannian feature preparation, one of which there exists 18 of.
    '''

    n_freq = len(model_dict['riemannian']['filter_bank'])
    max_error = 0
    mean_error = 0
    for freq_idx in range(n_freq):
        block = block_class(model_dict, freq_idx)
        x = F.quantize_to_int(data[input_name], block.input_scale, block.input_n_bits)
        if not input_name == 'input_quant':
            x = x[freq_idx]
        y_acq = block(x)
        y_exp = F.quantize_to_int(data[output_name][freq_idx], block.output_scale,
                                  block.output_n_bits)
        result = _compare_result(y_exp, y_acq)
        result = result['1']
        max_error = max(max_error, result['max error'])
        mean_error += result['avg error'] / n_freq

    success = max_error <= TOLERANCE
    return {block_class.__name__: {'result': success,
                                   'max error': max_error,
                                   'avg error': mean_error}}


def _compare_result(y_exp, y_hat, test_index=1, tolerance=6, epsilon=1e-4):
    """
    The error is computed in the following way:
    1. Scale the acquired output y_hat back into regular floating point representation
    2. take the maximal absolute difference e = max(|y_hat - y_exp|)
    3. Scale the error such that 1 represents an entire quantization step.
    4. We consider the output to be correct if this maximal scaled error is less than 1

    Parameters:
    - y_exp: np.array(type=int), expected result in integer representation
    - y_hat: np.array(type=int), acquired result in integer representation
    - test_index: index used in the return dict
    - tolerance: number of quantization steps the output is allowed to differ from the expected one.
    - epsilon: numerical stability

    Returns: dictionary: {test_index: {"result": bool, "max_error": int, "mean_error": float}}
    """
    abs_diff = np.abs(y_hat - y_exp)
    max_err = np.max(abs_diff)
    mean_err = np.mean(abs_diff)

    are_equal = max_err <= (tolerance + epsilon)

    return {str(test_index): {"result": are_equal, "max error": int(round(max_err)),
                              "avg error": mean_err}}
