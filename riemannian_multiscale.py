#!/usr/bin/env python3

'''	Functions used for calculating the Riemannian features'''

import numpy as np
from pyriemann.utils import mean, base
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

from filters import butter_fir_filter
from utils import quantize
from svd import logm
from sos_filt import quant_sos_filt, prepare_quant_filter

__author__ = "Michael Hersche and Tino Rellstab"
__email__ = "herschmi@ethz.ch,tinor@ethz.ch"


class RiemannianMultiscale:
    """ Riemannian feature multiscale class

    Parameters
    ----------

    filter_bank : array, shape (n_freq,order,(order))
                  Filterbank coefficients: If FIR dim = 2
                                           If IIR dim = 3

    temp_windows : array, shape (n_temp,2)
                   start and end sample of temporal window

    riem_opt: String {'Riemann', "Riemann_Euclid", "Whitened_Euclid", "No_Adaptation"}
              Riemannian option

    rho: float
         Regularization parameter for covariance calculation

    vectorized: bool
                Concatenate all frequency bands and temp window features to one vector

    Attributes
    ----------

    Examples
    --------

    """

    def __init__(self, filter_bank, temp_windows, riem_opt='Riemann', rho=0.1, vectorized=True):
        assert riem_opt in ["Riemann", "Riemann_Euclid", "Whitened_Euclid", "No_Adaptation"]
        # Frequency bands
        self.filter_bank = filter_bank
        self.n_freq = filter_bank.shape[0]
        # Temporal windows
        self.temp_windows = temp_windows
        self.n_temp = temp_windows.shape[0]
        # determine kernel function
        if riem_opt == 'Whitened_Euclid':
            self.riem_kernel = self.whitened_kernel
        else:
            self.riem_kernel = self.log_whitened_kernel
        # determine mean metric
        if riem_opt in ['Riemann', "Whitened_Diaglog"]:
            self.mean_metric = 'riemann'
        elif riem_opt in ['Riemann_Euclid', 'Whitened_Euclid']:
            self.mean_metric = 'euclid'
        self.riem_opt = riem_opt
        self.use_par = True

        # regularization
        self.rho = rho
        # vectorization (for SVM)
        self.vectorized = vectorized
        # initialize all other parameters
        self.n_channel = None
        self.n_riemann = None
        self.c_ref_invsqrtm = None

    def fit(self, data):
        '''
        Calculate average covariance matrices and return freatures of training data

        Parameters
        ----------
        data: array, shape (n_tr_trial,n_channel,n_samples)
              input training time samples

        Return
        ------
        train_feat: array, shape: if vectorized: (n_tr_trial,(n_temp x n_freq x n_riemann)
                                  else:          (n_tr_trial,n_temp , n_freq , n_riemann)
        '''

        n_tr_trial, n_channel, _ = data.shape
        self.n_channel = n_channel
        self.n_riemann = int((n_channel + 1) * n_channel / 2)

        if self.use_par:
            with Pool() as p:
                cov_mat = np.array(list(tqdm(p.imap(self._par_cov_mat, data),
                                             desc="Compute Covariance matrices", total=n_tr_trial,
                                             leave=False)))
        else:
            cov_mat = np.array(list(tqdm(map(self._par_cov_mat, data),
                                         desc="Compute Covariance matrices", total=n_tr_trial,
                                         leave=False)))

        # calculate mean covariance matrix
        self.c_ref_invsqrtm = np.zeros((self.n_freq, n_channel, n_channel))
        for freq_idx in tqdm(list(range(self.n_freq)), desc="Compute mean covariance matrix", leave=False):
            if self.riem_opt == 'No_Adaptation':
                self.c_ref_invsqrtm[freq_idx] = np.eye(n_channel)
            else:
                # Mean covariance matrix over all trials and temp winds per frequency band
                cov_avg = mean.mean_covariance(cov_mat[:, :, freq_idx].reshape(-1, n_channel, n_channel),
                                               metric=self.mean_metric)
                self.c_ref_invsqrtm[freq_idx] = base.invsqrtm(cov_avg)

        # calculate training features
        if self.use_par:
            with Pool() as p:
                train_feat = np.array(list(tqdm(p.imap(self._par_riemannian, cov_mat),
                                                desc="Compute Riemannian kernel", total=n_tr_trial,
                                                leave=False)))
        else:
            train_feat = np.array(list(tqdm(map(self._par_riemannian, cov_mat),
                                            desc="Compute Riemannian kernel", total=n_tr_trial,
                                            leave=False)))

        if self.vectorized:
            return train_feat.reshape(n_tr_trial, -1)
        return train_feat

    def features(self, data):
        '''
        Generate multiscale Riemannian features

        Parameters
        ----------
        data: array, shape (n_trial,n_channel,n_samples)
              input time samples

        Return
        ------
        feat: array, shape: if vectorized: (n_trial,(n_temp x n_freq x n_riemann)
                            else           (n_trial,n_temp , n_freq , n_riemann)
        '''
        n_trial = data.shape[0]

        feat = np.zeros((n_trial, self.n_temp, self.n_freq, self.n_riemann))

        with Pool() as p:
            cov_mat = np.array(list(tqdm(p.imap(self._par_cov_mat, data),
                                         desc="Compute Covariance matrices", total=n_trial,
                                         leave=False)))
            feat = np.array(list(tqdm(p.imap(self._par_riemannian, cov_mat),
                                      desc="Compute Riemannian kernel", total=n_trial,
                                      leave=False)))

        if self.vectorized:
            return feat.reshape(n_trial, -1)
        return feat

    def onetrial_feature(self, data):
        '''
        Generate multiscale Riemannian one trial and temp window

        Parameters
        ----------
        data: array, shape (n_channel,n_samples)
                input time samples

        Return
        ------
        feat: array, shape: if vectorized: (n_freq x n_riemann)
                            else           (n_freq , n_riemann)
        '''
        feat = np.zeros((self.n_freq, self.n_riemann))

        for freq_idx in range(self.n_freq):
            # filter signal
            data_filter = self._filter_signal(data, freq_idx)

            # regularized covariance matrix
            cov_mat = self._reg_cov_mat(data_filter, freq_idx)

            feat[freq_idx] = self.riem_kernel(cov_mat, self.c_ref_invsqrtm[freq_idx])

        if self.vectorized:
            return feat.reshape(-1)
        return feat

    def half_vectorization(self, mat):
        ''' Calculates half vectorization of a matrix

        Parameters
        ----------
        mat: array, shape(n_channel,n_channel)
             Input symmetric matrix

        Output
        ----------
        vec: array, shape (n_riemann,)
             Vectorized matrix
        '''
        _, N = mat.shape

        no_elements = ((N + 1) * N / 2)
        no_elements = int(no_elements)
        out_vec = np.zeros(no_elements)

        # fill diagonal elements with factor one
        for diag in range(0, N):
            out_vec[diag] = mat[diag, diag]

        sqrt2 = np.sqrt(2)
        idx = N
        for col in range(1, N):
            for row in range(0, col):
                out_vec[idx] = sqrt2 * mat[row, col]
                idx += 1
        return out_vec

    def _filter_signal(self, data, freq_idx):
        """ Apply the selected filter to the data """
        return butter_fir_filter(data, self.filter_bank[freq_idx])

    def _reg_cov_mat(self, data, freq_idx):
        """ Compute the regularized covariance matrix """
        n_samples = data.shape[1]
        n_channel = data.shape[0]
        return 1 / (n_samples - 1) * np.dot(data, np.transpose(data)) + self.rho / n_samples * np.eye(n_channel)

    def whitened_kernel(self, mat, c_ref_invsqrtm):
        return self.half_vectorization(np.dot(np.dot(c_ref_invsqrtm, mat), c_ref_invsqrtm))

    def log_whitened_kernel(self, mat, c_ref_invsqrtm):
        return self.half_vectorization(base.logm(np.dot(np.dot(c_ref_invsqrtm, mat), c_ref_invsqrtm)))

    def _par_cov_mat(self, data):
        """ function to compute the covariance matrix for a single sample """
        cov_mat = np.zeros((self.n_temp, self.n_freq, self.n_channel, self.n_channel))
        for temp_idx in range(self.n_temp):
            t_start, t_end = self.temp_windows[temp_idx, 0], self.temp_windows[temp_idx, 1]
            for freq_idx in range(self.n_freq):
                # filter signal
                data_filter = self._filter_signal(data[:, t_start:t_end], freq_idx)
                # regularized covariance matrix
                cov_mat[temp_idx, freq_idx] = self._reg_cov_mat(data_filter, freq_idx)
        return cov_mat

    def _par_riemannian(self, data):
        """ function to compute a the riemann kernel for a single sample """
        train_feat = np.zeros((self.n_temp, self.n_freq, self.n_riemann))
        for temp_idx in range(self.n_temp):
            for freq_idx in range(self.n_freq):
                train_feat[temp_idx, freq_idx] = self.riem_kernel(data[temp_idx, freq_idx],
                                                                  self.c_ref_invsqrtm[freq_idx])
        return train_feat


class QuantizedRiemannianMultiscale(RiemannianMultiscale):
    """ Quantized Riemannian feature multiscale class

    Parameters
    ----------

    filter_bank : array, shape (n_freq,order,(order))
                  Filterbank coefficients: If FIR dim = 2
                                           If IIR dim = 3

    temp_windows : array, shape (n_temp,2)
                   start and end sample of temporal window

    riem_opt: String {'Riemann', "Riemann_Euclid","Whitened_Euclid","No_Adaptation"}
              Riemannian option

    rho: float
         Regularization parameter for covariance calculation

    vectorized: bool
                Concatenate all frequency bands and temp window features to one vector

    num_bits: int
              Number of bits for quantization

    bitshift_scale: bool
                    If true, will make sure that bitshift can be used to transform from one layer to another

    """
    def __init__(self, filter_bank, temp_windows, riem_opt="Riemann", rho=0.1, vectorized=True,
                 num_bits=8, bitshift_scale=True):

        super(QuantizedRiemannianMultiscale, self).__init__(filter_bank, temp_windows, riem_opt=riem_opt,
                                                            rho=rho, vectorized=vectorized)
        self.num_bits = num_bits
        self.bitshift_scale = bitshift_scale

        self.scale_input = 0
        self.scale_filter_out = np.zeros((self.n_freq, ))
        self.scale_cov_mat = np.zeros((self.n_freq, ))
        self.scale_logm_out = 0
        self.scale_features = 0
        self.quant_filter_bank = []

        self.min_eigv = 1e10

        # filters must be second order sections
        assert self.filter_bank.shape[2] == 6

        # Go into monitor_ranges mode
        self.monitor_ranges = True

    def fit(self, data):
        assert not self.monitor_ranges, "Call \"self.determine_range\" before fitting the data"
        features = super(QuantizedRiemannianMultiscale, self).fit(data)
        features = self._output_quant(features)
        return features

    def features(self, data):
        assert not self.monitor_ranges, "Call \"self.determine_range\" before using \"self.features\""
        features = super(QuantizedRiemannianMultiscale, self).features(data)
        features = self._output_quant(features)
        return features

    def onetrial_feature(self, data):
        assert not self.monitor_ranges, "Call \"self.determine_range\" before using \"self.onetrial_feature\""
        features = super(QuantizedRiemannianMultiscale, self).onetrial_feature(data)
        features = self._output_quant(features)
        return features

    def prepare_quantization(self, data):
        """ Determine all scale ranges of the network and quantize the filters """
        # set the flag to monitor the ranges to true
        self.monitor_ranges = True
        self.use_par = False

        # call fit
        features = super(QuantizedRiemannianMultiscale, self).fit(data)
        features = self._output_quant(features)

        if self.bitshift_scale:
            # make sure that the transformation after the fitler is a power of two
            for band in range(self.filter_bank.shape[0]):
                # Formula for rescale factor: (Rx * Rw * Sy) / (Sx * Sw * Ry)
                # The ranges are already powers of two. (i.e. 8)
                # Sw is also a power of two
                # Thus, Sy / Sx must be a power of two. Since Sx is fixed, we need to change Sy.
                # Sy = Sx * 2^k, k in Z, k = ceil(log2(Sy_prev / Sx))
                k = int(np.ceil(np.log2(self.scale_filter_out[band] / self.scale_input)))
                self.scale_filter_out[band] = self.scale_input * (2 ** k)

        # prepare the filter quantization
        for band in range(self.filter_bank.shape[0]):
            self.quant_filter_bank.append(prepare_quant_filter(self.filter_bank[band],
                                                               self.scale_input,
                                                               self.scale_filter_out[band]))

        # set the flag to monitor the range to false, the modul can now be used
        self.use_par = True
        self.monitor_ranges = False

    def _quantize(self, data, factor, do_round=False, num_bits=None):
        """ Quantize the data to the given number of levels """
        if num_bits is None:
            num_bits = self.num_bits
        return quantize(data, factor, num_bits=num_bits, do_round=do_round)

    def _filter_signal(self, data, freq_idx):
        """ Apply the selected filter to the data """
        # measure the input scale or quantize
        if self.monitor_ranges:
            self.scale_input = max(self.scale_input, np.abs(data).max())
        else:
            # since this is the input, we can apply correct rounding.
            data = self._quantize(data, self.scale_input, do_round=True)

        # apply the filter
        if self.monitor_ranges:
            output = butter_fir_filter(data, self.filter_bank[freq_idx])
        else:
            output = np.zeros_like(data)
            for ch in range(data.shape[0]):
                output[ch] = quant_sos_filt(data[ch], self.quant_filter_bank[freq_idx],
                                            self.scale_input, self.scale_filter_out[freq_idx])
        # assert not np.any(np.isnan(output))

        # measure the output scale or quantize
        # bitshift scale will be applied at the end of the monitoring step
        if self.monitor_ranges:
            self.scale_filter_out[freq_idx] = max(self.scale_filter_out[freq_idx], np.abs(output).max())
        else:
            output = self._quantize(output, self.scale_filter_out[freq_idx])

        return output

    def _reg_cov_mat(self, data, freq_idx):
        """ Compute the regularized covariance matrix """
        n_samples = data.shape[1]
        n_channel = data.shape[0]

        mul_result = np.dot(data, np.transpose(data))

        cov_mat = 1/(n_samples-1) * mul_result + self.rho/n_samples*np.eye(n_channel)

        return cov_mat

    def _output_quant(self, data):
        """ quantization at the output of the feature extraction """
        if self.monitor_ranges:
            self.scale_features = max(self.scale_features, np.abs(data).max())
        else:
            data = self._quantize(data, self.scale_features)
        return data

    #def whitened_kernel(self, mat, c_ref_invsqrtm):
    #    return self.half_vectorization(np.dot(np.dot(c_ref_invsqrtm, mat), c_ref_invsqrtm))

    def log_whitened_kernel(self, mat, c_ref_invsqrtm):
        #return self.half_vectorization(base.logm(np.dot(np.dot(c_ref_invsqrtm, mat), c_ref_invsqrtm)))
        tmp = np.dot(np.dot(c_ref_invsqrtm, mat), c_ref_invsqrtm)
        # mat_log = base.logm(tmp)
        mat_log = logm(tmp)
        return self.half_vectorization(mat_log)
