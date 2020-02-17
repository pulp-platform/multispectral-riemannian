#!/usr/bin/env python3

''' 
Golden Model of the multiscale riemannian classifier.
'''

import numpy as np
import pickle
from functools import reduce
import operator

from functional import quantize, quantize_to_int, dequantize, quantize_iir_filter
from functional import prepare_bitshift, apply_bitshift_scale, solve_for_scale, solve_for_scale_sqr
from sos_filt import quant_sos_filt_df1
from svd import logm

__author__ = 'Tibor Schneider'
__email__ = 'sctibor@ethz.ch'


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


class GoldenModel:
    '''
    Fix-Point model for the multiscale riemannian classifier
    '''
    def __init__(self, model_filename):
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
        assert model_dict['riemannian']['c_ref_invsqrtm_n_bits'].shape[0] == self.n_freq
        assert model_dict['riemannian']['cov_mat_scale'].shape[0] == self.n_freq
        assert model_dict['riemannian']['cov_mat_n_bits'].shape[0] == self.n_freq
        assert model_dict['SVM']['weights'].shape[0] == 4
        assert model_dict['SVM']['bias'].shape[0] == 4

        self.temp_window = model_dict['riemannian']['temp_windows'][0]

        self.feature_extraction = FeatureExtraction(model_dict)
        self.svm = SVM(model_dict)

        self.input_shape = self.feature_extraction.input_shape
        self.input_scale = self.feature_extraction.input_shape
        self.input_n_bits = self.feature_extraction.input_n_bits
        self.output_shape = self.svm.output_shape
        self.output_scale = self.svm.output_shape
        self.output_n_bits = self.svm.output_n_bits

    def prepare_input(self, x):
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

    def __call__(self, x):
        return self.apply(x)

    def apply(self, x):
        ''' Do the inference '''
        assert x.shape == self.input_shape
        assert x.dtype == np.int

        features = self.featrue_extraction(x)
        y = self.svm(features)

        assert y.shape == self.output_shape
        assert y.dtype == np.int
        return y

    def dequantize_output(self, y):
        ''' Dequantize the output from the inference '''
        return dequantize(y, self.output_scale, self.output_n_bits)

    def apply_float(self, x):
        ''' apply the inference, where input and output are in floating point represenation '''
        x_quant = self.prepare_input(x)
        y_quant = self.apply(x_quant)
        y = self.dequantize_output(y_quant)
        return y


class Block():
    ''' Abstract class for a single block '''
    def __init__(self, model_dict):
        self.input_shape = None
        self.output_shape = None
        self.input_scale = 1
        self.output_scale = 1
        self.input_n_bits = model_dict['num_bits']
        self.output_n_bits = model_dict['num_bits']
        temp_window = model_dict['riemannian']['temp_windows'][0]
        self.T = temp_window[1] - temp_window[0]
        self.C = 22

    def apply(self, x):
        raise NotImplementedError()

    def __call__(self, x):
        return self.apply(x)


class SVM(Block):
    ''' SVM block '''
    def __init__(self, model_dict):
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
        self.weight = quantize(self.weight, self.weight_scale, self.weight_n_bits, True)
        assert self.weight.shape == (4, *self.input_shape)
        self.bias = model_dict['SVM']['bias']
        self.bias = quantize(self.bias, self.bias_scale, self.bias_n_bits, True)

    def apply(self, x):
        assert x.shape == self.input_shape
        assert x.dtype == np.int

        y = np.array([w @ x for w in self.weight]) + self.bias

        assert y.shape == self.output_shape
        assert y.dtype == np.int
        return y


class FeatureExtraction(Block):
    ''' Block containing the entire feature extraction '''
    def __init__(self, model_dict):
        super(FeatureExtraction, self).__init__(model_dict)

        self.n_freq = len(model_dict['riemannian']['filter_bank'])

        self.freq_band = [RiemannianFeature(model_dict, freq_idx)
                          for freq_idx in range(self.n_freq)]

        self.input_shape = (self.n_freq, *self.freq_band[0].input_shape)
        self.input_scale = self.freq_band[0].input_scale
        self.input_n_bits = self.freq_band[0].input_n_bits
        self.output_shape = (prod((self.n_freq, *self.freq_band[0].output_shape)), )
        self.output_scale = self.freq_band[0].output_scale
        self.output_n_bits = self.freq_band[0].output_n_bits

    def apply(self, x):
        assert x.shape == self.input_shape
        assert x.dtype == np.int

        y = np.array([feature(x) for feature in self.freq_band]).ravel()

        assert y.shape == self.output_shape
        assert y.dtype == np.int
        return y


class RiemannianFeature(Block):
    ''' Block containing feature preparation for a single frequency band '''
    def __init__(self, model_dict, freq_idx):
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

    def apply(self, x):
        x = self.filter(x)
        x = self.cov_mat(x)
        x = self.whitening(x)
        x = self.logm(x)
        x = self.half_diag(x)
        return x


class Filter(Block):
    ''' IIR filter block '''
    def __init__(self, model_dict, freq_idx):
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

    def apply(self, x):
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


class CovMat(Block):
    ''' Covariance Matrix Block '''
    def __init__(self, model_dict, freq_idx):
        super(CovMat, self).__init__(model_dict)
        self.freq_idx = freq_idx
        self.input_shape = (self.C, self.T)
        self.output_shape = (self.C, self.C)
        self.input_scale = model_dict['riemannian']['filter_out_scale'][freq_idx]
        self.output_scale = model_dict['riemannian']['cov_mat_scale'][freq_idx]
        self.output_n_bits = model_dict['riemannian']['cov_mat_n_bits']
        self.rho = model_dict['riemannian']['cov_mat_rho']
        self.rho = quantize_to_int(self.rho, self.output_scale, self.output_n_bits)

        # compute bitshift scale
        self.bitshift_scale = prepare_bitshift(self.input_scale, self.input_n_bits,
                                               self.input_scale, self.input_n_bits,
                                               self.output_scale, self.output_n_bits)

    def apply(self, x):
        assert x.shape == self.input_shape
        assert x.dtype == np.int

        y = x @ x.T
        y = apply_bitshift_scale(y, self.bitshift_scale, True)

        y = y + np.eye(self.C).astype(int) * self.rho

        assert y.shape == self.output_shape
        assert y.dtype == np.int
        return y


class Whitening(Block):
    ''' Whitening Block '''
    def __init__(self, model_dict, freq_idx):
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
        # quantize ref_invsqrtm to integer
        self.ref_invsqrtm = quantize_to_int(model_dict['riemannian']['c_ref_invsqrtm'][freq_idx],
                                            self.ref_invsqrtm_scale, self.ref_invsqrtm_n_bits)

    def apply(self, x):
        assert x.shape == self.input_shape
        assert x.dtype == np.int

        y = self.ref_invsqrtm @ x @ self.ref_invsqrtm

        if y.min() < -(1 << (self.output_n_bits - 1)) or \
           y.max() > (1 << (self.output_n_bits - 1)) - 1:
            raise OverflowError()

        assert y.shape == self.output_shape
        assert y.dtype == np.int
        return y


class Logm(Block):
    """ Applies matrix logarithm """
    def __init__(self, model_dict, freq_idx):
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

    def apply(self, x):
        assert x.shape == self.input_shape
        assert x.dtype == np.int

        x_float = dequantize(x, self.input_scale, self.input_n_bits)
        y_float = logm(x_float)
        y = quantize_to_int(y_float, self.output_scale, self.output_n_bits)

        assert y.shape == self.output_shape
        assert y.dtype == np.int
        return y


class HalfDiag(Block):
    """ Applies half-diagonalization """
    def __init__(self, model_dict, freq_idx):
        super(HalfDiag, self).__init__(model_dict)
        self.freq_idx = freq_idx
        self.input_shape = (self.C, self.C)
        self.output_shape = ((self.C * (self.C + 1)) // 2, )

        # determine input scale
        self.input_scale = model_dict['riemannian']['logm_out_scale']
        # self.input_n_bits = 8 # set by default
        self.output_scale = model_dict['riemannian']['features_scale']
        # self.output_n_bits = 8 # set by default

        self.bitshift_scale = prepare_bitshift(self.input_scale, self.input_n_bits,
                                               self.input_scale, self.input_n_bits,
                                               self.output_scale, self.output_n_bits)

        self.sqrt2 = quantize_to_int(np.sqrt(2), self.input_scale, self.input_n_bits)

    def apply(self, x):
        assert x.shape == self.input_shape
        assert x.dtype == np.int

        y = np.zeros(self.output_shape).astype(int)

        # first, fill in the diagonal elements
        y[:self.C] = np.diag(x)

        # now, fill in the remaining elements
        idx = self.C
        for col in range(1, self.C):
            for row in range(col):
                y[idx] = x[col, row] * self.sqrt2
                idx += 1

        # scale everything back
        apply_bitshift_scale(y, self.bitshift_scale)

        assert y.shape == self.output_shape
        assert y.dtype == np.int
        return y
