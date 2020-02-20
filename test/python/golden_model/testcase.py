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

np.set_printoptions(linewidth=200)

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
                                        'filter_out', 'cov_mat_reg_quant'))
    result.update(test_riemannian_block(model_dict, data, Whitening,
                                        'cov_mat_reg_quant', 'cov_mat_transform'))
    result.update(test_riemannian_block(model_dict, data, Logm,
                                        'cov_mat_transform', 'cov_mat_logm'))
    result.update(test_half_diag(model_dict, data, 'cov_mat_logm', 'features_quant'))
    result.update(test_riemannian_feature(model_dict, data, 'input_quant', 'features_quant'))
    result.update(test_feature_extraction(model_dict, data, 'input_quant', 'features_quant'))
    result.update(test_svm(model_dict, data, 'features_quant', 'svm_result'))
    result.update(test_golden_model(data, 'input_quant', 'svm_result'))

    logger.show_subcase_result('Individual Block:', result)

    # return summary
    return logger.summary()


def test_golden_model(data, input_name, output_name):
    '''
    Test golden model
    '''
    block = GoldenModel(MODEL_FILENAME)
    x = F.quantize_to_int(data[input_name], block.input_scale, block.input_n_bits, do_round=True)
    y_acq = block(x)
    y_exp = F.quantize_to_int(data[output_name], block.output_scale, block.output_n_bits)
    # shift the results back by 8 bits
    y_acq = F.apply_bitshift_scale(y_acq, 9)
    y_exp = F.apply_bitshift_scale(y_exp, 9)
    result = _compare_result(y_exp, y_acq)
    result = result['1']
    max_error = result['max error']
    mean_error = result['avg error']

    success = max_error <= TOLERANCE
    return {GoldenModel.__name__: {'result': success,
                                   'max error': max_error,
                                   'avg error': mean_error}}


def test_svm(model_dict, data, input_name, output_name):
    '''
    Test SVM
    '''
    block = SVM(model_dict)
    x = F.quantize_to_int(data[input_name], block.input_scale, block.input_n_bits, do_round=True)
    y_acq = block(x)
    y_exp = F.quantize_to_int(data[output_name], block.output_scale, block.output_n_bits)
    result = _compare_result(y_exp, y_acq)
    result = result['1']
    max_error = result['max error']
    mean_error = result['avg error']

    success = max_error <= TOLERANCE
    return {SVM.__name__: {'result': success,
                           'max error': max_error,
                           'avg error': mean_error}}

def test_feature_extraction(model_dict, data, input_name, output_name):
    '''
    Test FeatureExtraction class
    '''
    block = FeatureExtraction(model_dict)
    x = F.quantize_to_int(data[input_name], block.input_scale, block.input_n_bits, do_round=True)
    y_acq = block(x)
    y_exp = F.quantize_to_int(data[output_name], block.output_scale, block.output_n_bits)
    result = _compare_result(y_exp, y_acq)
    result = result['1']
    max_error = result['max error']
    mean_error = result['avg error']

    success = max_error <= TOLERANCE
    return {FeatureExtraction.__name__: {'result': success,
                                         'max error': max_error,
                                         'avg error': mean_error}}


def test_riemannian_feature(model_dict, data, input_name, output_name):
    '''
    Test RiemannianFeature class
    '''
    n_freq = len(model_dict['riemannian']['filter_bank'])
    max_error = 0
    mean_error = 0
    for freq_idx in range(n_freq):
        block = RiemannianFeature(model_dict, freq_idx)
        x = F.quantize_to_int(data[input_name], block.input_scale, block.input_n_bits, do_round=True)
        y_acq = block(x)
        y_exp = F.quantize_to_int(data[output_name], block.output_scale, block.output_n_bits)
        y_exp = y_exp[len(y_acq) * freq_idx:len(y_acq) * (freq_idx + 1)]
        result = _compare_result(y_exp, y_acq)
        result = result['1']
        max_error = max(max_error, result['max error'])
        mean_error += result['avg error'] / n_freq

    success = max_error <= TOLERANCE
    return {RiemannianFeature.__name__: {'result': success,
                                'max error': max_error,
                                'avg error': mean_error}}


def test_half_diag(model_dict, data, input_name, output_name):
    '''
    Test HalfDiag class
    '''
    n_freq = len(model_dict['riemannian']['filter_bank'])
    max_error = 0
    mean_error = 0
    for freq_idx in range(n_freq):
        block = HalfDiag(model_dict, freq_idx)
        x = F.quantize_to_int(data[input_name][freq_idx], block.input_scale, block.input_n_bits, do_round=True)
        y_acq = block(x)
        y_exp = F.quantize_to_int(data[output_name], block.output_scale, block.output_n_bits)
        y_exp = y_exp[len(y_acq) * freq_idx:len(y_acq) * (freq_idx + 1)]
        result = _compare_result(y_exp, y_acq)
        result = result['1']
        max_error = max(max_error, result['max error'])
        mean_error += result['avg error'] / n_freq

    success = max_error <= TOLERANCE
    return {HalfDiag.__name__: {'result': success,
                                'max error': max_error,
                                'avg error': mean_error}}


def test_riemannian_block(model_dict, data, block_class, input_name, output_name):
    '''
    Test a block inside the riemannian feature preparation, one of which there exists 18 of.
    '''

    n_freq = len(model_dict['riemannian']['filter_bank'])
    max_error = 0
    mean_error = 0
    for freq_idx in range(n_freq):
        block = block_class(model_dict, freq_idx)
        x = F.quantize_to_int(data[input_name], block.input_scale, block.input_n_bits, do_round=True)
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


def _compare_result(y_exp, y_hat, test_index=1, tolerance=1, epsilon=1e-4):
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
