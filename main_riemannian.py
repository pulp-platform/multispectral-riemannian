#!/usr/bin/env python3

''' 
Model for Riemannian feature calculation and classification for EEG data
'''

import time
import numpy as np

from sklearn.model_selection import KFold

# import self defined functions
from riemannian_model import RiemannianModel, QuantizedRiemannianModel
from get_data import get_data

__author__ = "Michael Hersche and Tino Rellstab"
__email__ = "herschmi@ethz.ch,tinor@ethz.ch"

DATA_PATH = "dataset/"
QUANTIZED = True
BANDS = [2] # [2, 4, 8, 16, 32]
CROSS_VALIDATION = False
CV_NO_SPLITS = 5
NO_SUBJECTS = 9
RANDOM_SEED = 1 # None


def main():
    """ main function """

    if QUANTIZED:
        model = QuantizedRiemannianModel(bands=BANDS, random_state=RANDOM_SEED)
    else:
        model = RiemannianModel(bands=BANDS, random_state=RANDOM_SEED)


    # success rate sum over all subjects
    success_tot_sum = 0

    if CROSS_VALIDATION:
        print("Cross validation run")
    else:
        print("Test data set")

    start = time.time()

    # Go through all subjects
    for subject in range(1, NO_SUBJECTS + 1):

        # load data
        samples, labels = get_data(subject, True, DATA_PATH)

        if CROSS_VALIDATION:
            success_sub_sum = 0

            for train_index, val_index in KFold(CV_NO_SPLITS).split(samples):
                model.fit(samples[train_index], labels[train_index])
                success_sub_sum += model.score(samples[val_index], labels[val_index])

            # average over all splits
            success_rate = success_sub_sum / CV_NO_SPLITS

        else:
            test_samples, test_labels = get_data(subject, False, DATA_PATH)
            # load Eval data
            model.fit(samples, labels)
            success_rate = model.score(test_samples, test_labels)

        print(f"Subject {subject}: {success_rate}")

        success_tot_sum += success_rate

    end = time.time()

    # Average success rate over all subjects
    print("Average success rate: " + str(success_tot_sum / NO_SUBJECTS))
    print("Time elapsed [s] " + str(end - start))


if __name__ == '__main__':
    main()
