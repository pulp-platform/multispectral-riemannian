#!/usr/bin/env python3

''' 
Model for Riemannian feature calculation and classification for EEG data
'''

import numpy as np
from sklearn.svm import LinearSVC, SVC

from riemannian_multiscale import RiemannianMultiscale, QuantizedRiemannianMultiscale
from filters import load_filterbank
from utilities import quantize

__author__ = "Michael Hersche, Tino Rellstab and Tibor Schneider"
__email__ = "herschmi@ethz.ch,tinor@ethz.ch"

DATA_PATH = "dataset/"
QUANTIZED = True
# ONLY_2HZ_BANDS = True

class RiemannianModel():
    """ Riemannian Model """
    def __init__(self, svm_kernel='linear', svm_c=0.1, fs=250, bands=None, time_windows=None,
                 riem_opt='Riemann', rho=0.1, filter_type='butter', filter_order=2,
                 random_state=None):
        """ Constructor

        Args:

        Parameters
        ----------

        svm_kernel: str {'linear', 'sigmoid', 'rbf'}
                    kernel used for classifier

        svm_c: float
               regularization parameter for the classifier

        fs: int
            sampling rate of the data

        bands: list of int
                bandwidths used in filterbanks (default: [2, 4, 8, 16, 32])

        time_windows: list of list of ints, shape = (N, 2)
                      time windows used, in seconds (default: [[2,5, 6]])

        riem_opt: str {"riemann", "Riemann_Euclid", "Whitened_Euclid", "No_Adaptation"}
                  type of riemannian used

        rho: float
             Normalization parameter for the covariance matrix of the riemannian

        filter_type: str {"butter", "fir"}
                     Type of the filter

        filter_order: int
                      Order of the filter

        random_state: int or None
                      random seed used in the SVM
        """

        # setup classifier
        if svm_kernel == 'linear':
            self.classifier = LinearSVC(C=svm_c, loss='hinge', random_state=random_state, tol=0.00001)
        else:
            self.classifier = SVC(C=svm_c, kernel=svm_kernel, degree=10, gamma='auto',
                                  cache_size=10000, random_state=random_state)

        # setup Filterbank
        if bands is None:
            bandwidths = np.array([2, 4, 8, 16, 32])
        else:
            bandwidths = np.array(bands)
        filter_bank = load_filterbank(bandwidths, fs, order=filter_order, max_freq=40, ftype=filter_type)

        # setup Time Windows
        if time_windows is None:
            time_windows = (np.array([[2.5, 6]]) * fs).astype(int)
        else:
            time_windows = (np.array(time_windows) * fs).astype(int)

        # setup riemannian
        self.riemannian = RiemannianMultiscale(filter_bank, time_windows, riem_opt=riem_opt,
                                               rho=rho, vectorized=True)

        # store dimensionality
        self.no_bands = filter_bank.shape[0]
        self.no_time_windows = time_windows.shape[0]
        self.no_riem = None
        self.no_features = None

    def fit(self, samples, labels):
        """ Training

        Parameters
        ----------

        samples: np.array, size=(N, C, T)
                 training samples

        labels: np.array, size=(N)
                training labels
        """
        # extract the number of eatures
        assert len(samples.shape) == 3
        no_channels = samples.shape[1]
        self.no_riem = int(no_channels * (no_channels + 1) / 2)  # Total number of CSP feature per band and timewindow
        self.no_features = self.no_riem * self.no_bands * self.no_time_windows

        # fit and extract training features from the riemannian
        features = self.riemannian.fit(samples)
        self.classifier.fit(features, labels)

    def score(self, samples, labels):
        """ Measure the performance, returns success rate

        Parameters
        ----------

        samples: np.array, size=(N, C, T)
                 training samples

        labels: np.array, size=(N)
                training labels

        Returns
        -------

        float: score of the model
        """
        features = self.riemannian.features(samples)
        return self.classifier.score(features, labels)

    def predict(self, samples):
        """ Predict some data

        Parameters
        ----------

        samples: np.array, size=(N, C, T)
                 training samples

        Returns
        -------

        np.array, size=[N]: prediction
        """
        features = self.riemannian.features(samples)
        return self.classifier.predict(features)


class QuantizedRiemannianModel():
    """ QuantizedRiemannian Model """
    def __init__(self, svm_c=0.1, fs=250, bands=None, riem_opt='Riemann', rho=0.1, filter_order=2,
                 random_state=None, num_bits=8, bitshift_scale=True):
        """ Constructor

        Parameters
        ----------

        svm_c: float
               regularization parameter for the classifier

        fs: int
            sampling rate of the data

        bands: list of int
                bandwidths used in filterbanks (default: [2, 4, 8, 16, 32])

        riem_opt: str {"riemann", "Riemann_Euclid", "Whitened_Euclid", "No_Adaptation"}
                  type of riemannian used

        rho: float
             Normalization parameter for the covariance matrix of the riemannian

        filter_order: int
                      Order of the filter

        random_state: int or None
                      random seed used in the SVM

        num_bits: int
                  Number of bits used for quantization

        bitshift_scale: bool
                        if True, make sure that all scale factors between one part and the next is a bitshift
        """

        self.num_bits = num_bits
        self.bitshift_scale = bitshift_scale

        # setup classifier
        self.classifier = LinearSVC(C=svm_c, loss='hinge', random_state=random_state, tol=0.00001)

        # setup Filterbank
        if bands is None:
            bandwidths = np.array([2, 4, 8, 16, 32])
        else:
            bandwidths = np.array(bands)
        filter_bank = load_filterbank(bandwidths, fs, order=filter_order, max_freq=40, ftype="butter")

        # setup Time Windows
        time_windows = (np.array([[2.5, 6]]) * fs).astype(int)

        # setup riemannian
        self.riemannian = QuantizedRiemannianMultiscale(filter_bank, time_windows, riem_opt=riem_opt,
                                                        rho=rho, vectorized=True, num_bits=num_bits,
                                                        bitshift_scale=bitshift_scale)

        # prepare quantized weights and biases
        self.scale_weight = 0
        self.scale_bias = 0

        # store dimensionality
        self.no_bands = filter_bank.shape[0]
        self.no_time_windows = time_windows.shape[0]
        self.no_riem = None
        self.no_features = None

    def fit(self, samples, labels):
        """ Training

        Parameters
        ----------

        samples: np.array, size=(N, C, T)
                 training samples

        labels: np.array, size=(N)
                training labels
        """
        # extract the number of eatures
        assert len(samples.shape) == 3
        no_channels = samples.shape[1]
        self.no_riem = int(no_channels * (no_channels + 1) / 2)  # Total number of CSP feature per band and timewindow
        self.no_features = self.no_riem * self.no_bands * self.no_time_windows

        # prepare scale factors
        self.riemannian.prepare_quantization(samples)

        # fit and extract training features from the riemannian
        features = self.riemannian.fit(samples)
        self.classifier.fit(features, labels)

        # quantize the classifier
        self.scale_weight = max(self.scale_weight, np.abs(self.classifier.coef_).max())
        weights = quantize(self.classifier.coef_, self.scale_weight, self.num_bits, do_round=True)
        self.classifier.coef_ = weights

        # do not quantize the bias, this one will be added in 32 bit, and quantization does not
        # matter here...

        # self.scale_bias = max(self.scale_bias, np.abs(self.classifier.intercept_).max())
        # bias = quantize(self.classifier.intercept_, self.scale_weight, self.num_bits,
        #                 do_round=True)
        # self.classifier.intercept_ = bias

    def score(self, samples, labels):
        """ Measure the performance, returns success rate

        Parameters
        ----------

        samples: np.array, size=(N, C, T)
                 training samples

        labels: np.array, size=(N)
                training labels

        Returns
        -------

        float: score of the model
        """
        features = self.riemannian.features(samples)
        return self.classifier.score(features, labels)

    def predict(self, samples):
        """ Predict some data

        Parameters
        ----------

        samples: np.array, size=(N, C, T)
                 training samples

        Returns
        -------

        np.array, size=[N]: prediction
        """
        features = self.riemannian.features(samples)
        return self.classifier.predict(features)

    def predict_with_intermediate(self, sample, verbose=True):
        """ Predict some data

        Parameters
        ----------

        samples: np.array, size=(C, T)
                 training sample

        Returns
        -------

        ordered dictionary including every intermediate result and the output
        """
        if verbose:
            print("Predict sample with intermediate matrices")
        assert len(sample.shape) == 2
        result = self.riemannian.onetrial_feature_with_intermediate(sample)
        features = next(reversed(result.values()))
        features = features.reshape(1, -1)
        result["svm_result"] = self.classifier.decision_function(features)
        result["prediction"] = self.classifier.predict(features)
        return result

    def get_data_dict(self):
        """ Returns a nested dictionary containing all necessary data """
        return {"num_bits": self.num_bits,
                "bitshift_scale": self.bitshift_scale,
                "SVM": {"weights": self.classifier.coef_,
                        "weight_scale": self.scale_weight,
                        "bias": self.classifier.intercept_},
                "riemannian": self.riemannian.get_data_dict()}
