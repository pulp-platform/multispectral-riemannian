#!/usr/bin/env python3

''' 
Golden Model of the multiscale riemannian classifier.
'''

import numpy as np
import pickle
from functools import reduce
import operator
from collections import OrderedDict

from functional import quantize, quantize_to_int, dequantize, quantize_iir_filter
from functional import prepare_bitshift, apply_bitshift_scale, solve_for_scale, solve_for_scale_sqr
from sos_filt import quant_sos_filt_df1
from svd import logm
from header_file import HeaderFile, HeaderConstant, HeaderArray, HeaderScalar, HeaderComment, \
    HeaderStruct

__author__ = 'Tibor Schneider'
__email__ = 'sctibor@ethz.ch'


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


class GoldenModel:
    '''
    Fix-Point model for the multiscale riemannian classifier
    '''
    def __init__(self, model_filename: str) -> None:
        ''' Initialize the RiemannianItegerModel

        Parameters
        ----------
        model_filename: string
                        Filename (including path) to the model.pkl file, exported from
                        QuantizedRiemannianModel
        '''
        # load the model dict
        with open(model_filename, 'rb') as _f:
            model_dict = pickle.load(_f)

        self.n_freq = len(model_dict['riemannian']['filter_bank'])

        # check the model dict
        assert model_dict['riemannian']['riem_opt'] == 'Riemann'
        assert model_dict['bitshift_scale']
        assert len(model_dict['riemannian']['temp_windows']) == 1
        assert model_dict['riemannian']['filter_out_scale'].shape[0] == self.n_freq
        assert model_dict['riemannian']['c_ref_invsqrtm'].shape[0] == self.n_freq
        assert model_dict['riemannian']['c_ref_invsqrtm_scale'].shape[0] == self.n_freq
        assert model_dict['riemannian']['cov_mat_scale'].shape[0] == self.n_freq
        assert model_dict['SVM']['weights'].shape[0] == 4
        assert model_dict['SVM']['bias'].shape[0] == 4

        self.temp_window = model_dict['riemannian']['temp_windows'][0]

        self.feature_extraction = FeatureExtraction(model_dict)
        self.svm = SVM(model_dict)

        self.input_shape = self.feature_extraction.input_shape
        self.input_scale = self.feature_extraction.input_scale
        self.input_n_bits = self.feature_extraction.input_n_bits
        self.output_shape = self.svm.output_shape
        self.output_scale = self.svm.output_scale
        self.output_n_bits = self.svm.output_n_bits

    def prepare_input(self, x: np.ndarray) -> np.ndarray:
        ''' quantizes the input and uses the correct time window '''
        assert x.dtype in [np.float32, np.float64]

        if x.shape != self.input_shape:
            # use the correct time window
            t_start, t_end = self.temp_window
            x = x[:, t_start:t_end]

        # quantize the input
        x = quantize_to_int(x, self.input_scale, self.input_n_bits)

        assert x.shape == self.input_shape
        assert x.dtype == np.int
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.apply(x)

    def apply(self, x: np.ndarray) -> np.ndarray:
        ''' Do the inference '''
        assert x.shape == self.input_shape
        assert x.dtype == np.int

        features = self.feature_extraction(x)
        y = self.svm(features)

        assert y.shape == self.output_shape
        assert y.dtype == np.int
        return y

    def dequantize_output(self, y: np.ndarray) -> np.ndarray:
        ''' Dequantize the output from the inference '''
        return dequantize(y, self.output_scale, self.output_n_bits)

    def apply_float(self, x: np.ndarray) -> np.ndarray:
        ''' apply the inference, where input and output are in floating point represenation '''
        x_quant = self.prepare_input(x)
        y_quant = self.apply(x_quant)
        y = self.dequantize_output(y_quant)
        return y


class Block():
    ''' Abstract class for a single block '''
    def __init__(self, model_dict: dict) -> None:
        self.input_shape = None
        self.output_shape = None
        self.input_scale = 1
        self.output_scale = 1
        self.input_n_bits = model_dict['num_bits']
        self.output_n_bits = model_dict['num_bits']
        temp_window = model_dict['riemannian']['temp_windows'][0]
        self.T = temp_window[1] - temp_window[0]
        self.C = 22

    def apply(self, x: np.ndarray) -> np.ndarray:
        """ apply the function of the block to some input """
        raise NotImplementedError()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.apply(x)

    def add_header_entries(self, header_file: HeaderFile, base_name: str) -> None:
        """ Adds necessary header entries to the file """
        raise NotImplementedError()


class SVM(Block):
    ''' SVM block '''
    def __init__(self, model_dict: dict) -> None:
        super(SVM, self).__init__(model_dict)

        # compute the input shape
        n_freq = len(model_dict['riemannian']['filter_bank'])
        self.input_shape = (n_freq * (self.C * (self.C + 1) // 2), )
        self.input_scale = model_dict['riemannian']['features_scale']
        self.weight_scale = model_dict['SVM']['weight_scale']
        self.weight_n_bits = self.input_n_bits

        # determine the output scale
        self.output_n_bits = 32
        self.output_scale = solve_for_scale(self.input_scale, self.input_n_bits,
                                            self.weight_scale, self.weight_n_bits,
                                            self.output_n_bits)
        self.output_shape = (4, )

        # use the same scale number of bits as output for the bias
        self.bias_scale = self.output_scale
        self.bias_n_bits = self.output_n_bits

        # quantize the weights and the bias
        self.weight = model_dict['SVM']['weights']
        self.weight = quantize_to_int(self.weight, self.weight_scale, self.weight_n_bits, True)
        assert self.weight.shape == (4, *self.input_shape)
        self.bias = model_dict['SVM']['bias']
        self.bias = quantize_to_int(self.bias, self.bias_scale, self.bias_n_bits, True)

    def apply(self, x: np.ndarray) -> np.ndarray:
        assert x.shape == self.input_shape
        assert x.dtype == np.int

        y = np.array([w @ x for w in self.weight]) + self.bias

        assert y.shape == self.output_shape
        assert y.dtype == np.int
        assert all([-(1 << 31) <= t < (1 << 31) for t in y])
        return y

    def add_header_entries(self, header_file: HeaderFile, base_name: str) -> None:
        """ Adds necessary header entries to the file """
        raise NotImplementedError()


class FeatureExtraction(Block):
    ''' Block containing the entire feature extraction '''
    def __init__(self, model_dict: dict) -> None:
        super(FeatureExtraction, self).__init__(model_dict)

        self.n_freq = len(model_dict['riemannian']['filter_bank'])

        self.freq_band = [RiemannianFeature(model_dict, freq_idx)
                          for freq_idx in range(self.n_freq)]

        self.input_shape = self.freq_band[0].input_shape
        self.input_scale = self.freq_band[0].input_scale
        self.input_n_bits = self.freq_band[0].input_n_bits
        self.output_shape = (prod((self.n_freq, *self.freq_band[0].output_shape)), )
        self.output_scale = self.freq_band[0].output_scale
        self.output_n_bits = self.freq_band[0].output_n_bits

    def apply(self, x: np.ndarray) -> np.ndarray:
        assert x.shape == self.input_shape
        assert x.dtype == np.int

        y = np.array([feature(x) for feature in self.freq_band]).ravel()

        assert y.shape == self.output_shape
        assert y.dtype == np.int
        return y

    def add_header_entries(self, header_file: HeaderFile, base_name: str) -> None:
        """ Adds necessary header entries to the file """
        raise NotImplementedError()


class RiemannianFeature(Block):
    ''' Block containing feature preparation for a single frequency band '''
    def __init__(self, model_dict: dict, freq_idx: int) -> None:
        super(RiemannianFeature, self).__init__(model_dict)
        self.freq_idx = freq_idx

        self.filter = Filter(model_dict, freq_idx)
        self.cov_mat = CovMat(model_dict, freq_idx)
        self.whitening = Whitening(model_dict, freq_idx)
        self.logm = Logm(model_dict, freq_idx)
        self.half_diag = HalfDiag(model_dict, freq_idx)

        self.input_shape = self.filter.input_shape
        self.input_scale = self.filter.input_scale
        self.input_n_bits = self.filter.input_n_bits
        self.output_shape = self.half_diag.output_shape
        self.output_scale = self.half_diag.output_scale
        self.output_n_bits = self.half_diag.output_n_bits

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = self.filter(x)
        x = self.cov_mat(x)
        x = self.whitening(x)
        x = self.logm(x)
        x = self.half_diag(x)
        return x

    def add_header_entries(self, header_file: HeaderFile, base_name: str) -> None:
        """ Adds necessary header entries to the file """
        raise NotImplementedError()


class Filter(Block):
    ''' IIR filter block '''
    def __init__(self, model_dict: dict, freq_idx: int) -> None:
        super(Filter, self).__init__(model_dict)
        self.freq_idx = freq_idx
        self.input_shape = (self.C, self.T)
        self.output_shape = (self.C, self.T)
        self.input_scale = model_dict['riemannian']['input_scale']
        self.output_scale = model_dict['riemannian']['filter_out_scale'][freq_idx]
        quant_filter = model_dict['riemannian']['filter_bank'][freq_idx]
        filter_n_bits = model_dict['riemannian']['filter_n_bits']
        prep_filter = quantize_iir_filter(quant_filter, filter_n_bits)
        self.coeff_a, self.shift_a, self.coeff_b, self.shift_b, self.shift_y = prep_filter

    def apply(self, x: np.ndarray) -> np.ndarray:
        assert x.shape == self.input_shape
        assert x.dtype == np.int

        y = np.array([quant_sos_filt_df1(channel, self.coeff_a, self.coeff_b,
                                         self.shift_a, self.shift_b)
                      for channel in x])

        # shift y back
        if self.shift_y < 0:
            y = y >> -self.shift_y
        elif self.shift_y > 0:
            y = y << self.shift_y

        # throw away the last bits of the computation
        y = y[:, :x.shape[1]]

        assert y.shape == self.output_shape
        assert y.dtype == np.int

        return y

    def add_header_entries(self,
                           header_file: HeaderFile,
                           base_name: str,
                           is_full_name: bool = False) -> None:
        """ Adds necessary header entries to the file """
        if is_full_name:
            name = base_name
        else:
            name = f"{base_name}_filter_param_{self.freq_idx}"
        # First, add a comment
        header_file.add(HeaderComment(f"Filter Coefficients for frequency id: {self.freq_idx}",
                                      mode="//", blank_line=False))
        # add the struct
        struct = OrderedDict()
        struct['a01'] = f"((v2s){{ {self.coeff_a[1, 1]}, {self.coeff_a[1, 2]} }})"
        struct['a11'] = f"((v2s){{ {self.coeff_a[2, 1]}, {self.coeff_a[2, 2]} }})"
        struct['b00'] = f"{self.coeff_b[1, 0]}"
        struct['b01'] = f"((v2s){{ {self.coeff_b[1, 1]}, {self.coeff_b[1, 2]} }})"
        struct['b10'] = f"{self.coeff_b[2, 0]}"
        struct['b11'] = f"((v2s){{ {self.coeff_b[2, 1]}, {self.coeff_b[2, 2]} }})"
        struct['shift_a0'] = f"{self.shift_a[1]}"
        struct['shift_a1'] = f"{self.shift_a[2]}"
        struct['shift_b0'] = f"{self.shift_b[1]}"
        struct['shift_b1'] = f"{self.shift_b[2]}"
        struct['y_shift'] = f"{-self.shift_y}"
        header_file.add(HeaderStruct(name, "func_sos_filt_2S_params_t", struct))

    def get_initializer_str(self, double_tab=False):
        struct = OrderedDict()
        struct['a01'] = f"((v2s){{ {self.coeff_a[1, 1]}, {self.coeff_a[1, 2]} }})"
        struct['a11'] = f"((v2s){{ {self.coeff_a[2, 1]}, {self.coeff_a[2, 2]} }})"
        struct['b00'] = f"{self.coeff_b[1, 0]}"
        struct['b01'] = f"((v2s){{ {self.coeff_b[1, 1]}, {self.coeff_b[1, 2]} }})"
        struct['b10'] = f"{self.coeff_b[2, 0]}"
        struct['b11'] = f"((v2s){{ {self.coeff_b[2, 1]}, {self.coeff_b[2, 2]} }})"
        struct['shift_a0'] = f"{self.shift_a[1]}"
        struct['shift_a1'] = f"{self.shift_a[2]}"
        struct['shift_b0'] = f"{self.shift_b[1]}"
        struct['shift_b1'] = f"{self.shift_b[2]}"
        struct['y_shift'] = f"{-self.shift_y}"
        s = HeaderStruct("only_initializer_str", "func_sos_filt_2S_params_t", struct)
        return s.initializer_str(double_tab=double_tab)


class CovMat(Block):
    ''' Covariance Matrix Block '''
    def __init__(self, model_dict: dict, freq_idx: int) -> None:
        super(CovMat, self).__init__(model_dict)
        self.freq_idx = freq_idx
        self.input_shape = (self.C, self.T)
        self.output_shape = (self.C, self.C)
        self.input_scale = model_dict['riemannian']['filter_out_scale'][freq_idx]
        self.output_scale = model_dict['riemannian']['cov_mat_scale'][freq_idx]
        self.output_n_bits = model_dict['riemannian']['cov_mat_n_bits']

        self.intermediate_n_bits = 32
        self.intermediate_scale = solve_for_scale(self.input_scale, self.input_n_bits,
                                                  self.input_scale, self.input_n_bits,
                                                  self.intermediate_n_bits)

        self.rho = model_dict['riemannian']['cov_mat_rho']
        self.rho = quantize_to_int(self.rho, self.intermediate_scale, self.intermediate_n_bits)

        # compute bitshift scale
        self.bitshift_scale = prepare_bitshift(self.input_scale, self.input_n_bits,
                                               self.input_scale, self.input_n_bits,
                                               self.output_scale, self.output_n_bits)

    def apply(self, x: np.ndarray) -> np.ndarray:
        assert x.shape == self.input_shape
        assert x.dtype == np.int

        y = x @ x.T
        y = y + np.eye(self.C).astype(int) * self.rho

        y = apply_bitshift_scale(y, self.bitshift_scale, True)

        assert y.shape == self.output_shape
        assert y.dtype == np.int
        return y

    def add_header_entries(self, header_file: HeaderFile, base_name: str) -> None:
        """ Adds necessary header entries to the file """
        raise NotImplementedError()


class Whitening(Block):
    ''' Whitening Block '''
    def __init__(self, model_dict: dict, freq_idx: int) -> None:
        super(Whitening, self).__init__(model_dict)
        self.freq_idx = freq_idx
        self.input_shape = (self.C, self.C)
        self.output_shape = (self.C, self.C)
        self.input_scale = model_dict['riemannian']['cov_mat_scale'][freq_idx]
        self.input_n_bits = model_dict['riemannian']['cov_mat_n_bits']
        self.ref_invsqrtm_scale = model_dict['riemannian']['c_ref_invsqrtm_scale'][freq_idx]
        self.ref_invsqrtm_n_bits = model_dict['riemannian']['c_ref_invsqrtm_n_bits']

        # determine output scale
        self.output_n_bits = 32
        self.output_scale = solve_for_scale_sqr(self.input_scale, self.input_n_bits,
                                                self.ref_invsqrtm_scale, self.ref_invsqrtm_n_bits,
                                                self.output_n_bits)

        # check that c_ref is quantized
        c_ref_invsqrtm = model_dict['riemannian']['c_ref_invsqrtm'][freq_idx]
        np.testing.assert_almost_equal(c_ref_invsqrtm,
                                       quantize(c_ref_invsqrtm, self.ref_invsqrtm_scale,
                                                self.ref_invsqrtm_n_bits, do_round=True))

        # quantize ref_invsqrtm to integer
        self.ref_invsqrtm = quantize_to_int(c_ref_invsqrtm, self.ref_invsqrtm_scale,
                                            self.ref_invsqrtm_n_bits)

    def apply(self, x: np.ndarray) -> np.ndarray:
        assert x.shape == self.input_shape
        assert x.dtype == np.int

        y = self.ref_invsqrtm @ x @ self.ref_invsqrtm

        if y.min() < -(1 << (self.output_n_bits - 1)) or \
           y.max() > (1 << (self.output_n_bits - 1)) - 1:
            raise OverflowError()

        assert y.shape == self.output_shape
        assert y.dtype == np.int
        return y

    def add_header_entries(self, header_file: HeaderFile, base_name: str) -> None:
        """ Adds necessary header entries to the file """
        raise NotImplementedError()


class Logm(Block):
    """ Applies matrix logarithm """
    def __init__(self, model_dict: dict, freq_idx: int) -> None:
        super(Logm, self).__init__(model_dict)
        self.freq_idx = freq_idx
        self.input_shape = (self.C, self.C)
        self.output_shape = (self.C, self.C)

        # determine input scale
        cov_scale = model_dict['riemannian']['cov_mat_scale'][freq_idx]
        cov_n_bits = model_dict['riemannian']['cov_mat_n_bits']
        ref_scale = model_dict['riemannian']['c_ref_invsqrtm_scale'][freq_idx]
        ref_n_bits = model_dict['riemannian']['c_ref_invsqrtm_n_bits']
        self.input_n_bits = 32
        self.input_scale = solve_for_scale_sqr(cov_scale, cov_n_bits, ref_scale, ref_n_bits,
                                               self.input_n_bits)
        # self.output_n_bits = 8 # set by default
        self.output_scale = model_dict['riemannian']['logm_out_scale']

    def apply(self, x: np.ndarray) -> np.ndarray:
        assert x.shape == self.input_shape
        assert x.dtype == np.int

        x_float = dequantize(x, self.input_scale, self.input_n_bits)
        y_float = logm(x_float)
        y = quantize_to_int(y_float, self.output_scale, self.output_n_bits)

        assert y.shape == self.output_shape
        assert y.dtype == np.int
        return y

    def add_header_entries(self, header_file: HeaderFile, base_name: str) -> None:
        """ Adds necessary header entries to the file """
        raise NotImplementedError()


class HalfDiag(Block):
    """ Applies half-diagonalization """
    def __init__(self, model_dict: dict, freq_idx: int) -> None:
        super(HalfDiag, self).__init__(model_dict)
        self.freq_idx = freq_idx
        self.input_shape = (self.C, self.C)
        self.output_shape = ((self.C * (self.C + 1)) // 2, )

        # determine input scale
        self.input_scale = model_dict['riemannian']['logm_out_scale']
        # self.input_n_bits = 8 # set by default
        self.output_scale = model_dict['riemannian']['features_scale']
        # self.output_n_bits = 8 # set by default
        sqrt2_scale = 2
        sqrt2_n_bits = self.input_n_bits

        self.bitshift_scale = prepare_bitshift(self.input_scale, self.input_n_bits,
                                               sqrt2_scale, sqrt2_n_bits,
                                               self.output_scale, self.output_n_bits)

        self.bitshift_scale_diag = np.log2(self.output_scale / self.input_scale)
        np.testing.assert_almost_equal(self.bitshift_scale_diag,
                                       np.round(self.bitshift_scale_diag))
        self.bitshift_scale_diag = int(np.round(self.bitshift_scale_diag))

        self.sqrt2 = quantize_to_int(np.sqrt(2), sqrt2_scale, sqrt2_n_bits)

    def apply(self, x: np.ndarray) -> np.ndarray:
        assert x.shape == self.input_shape
        assert x.dtype == np.int

        y = np.zeros(self.output_shape).astype(int)

        # first, fill in the diagonal elements
        y[:self.C] = apply_bitshift_scale(np.diag(x), self.bitshift_scale_diag)

        # now, fill in the remaining elements
        idx = self.C
        for col in range(1, self.C):
            for row in range(col):
                y[idx] = x[col, row] * self.sqrt2
                idx += 1

        y[self.C:] = apply_bitshift_scale(y[self.C:], self.bitshift_scale)

        assert y.shape == self.output_shape
        assert y.dtype == np.int
        return y

    def add_header_entries(self, header_file: HeaderFile, base_name: str) -> None:
        """ Adds necessary header entries to the file """
        raise NotImplementedError()
