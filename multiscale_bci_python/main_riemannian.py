#!/usr/bin/env python3

''' 
Model for Riemannian feature calculation and classification for EEG data
'''

# hide all sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import time
import os
import pickle
import numpy as np
import argparse

from tqdm import tqdm
from sklearn.model_selection import KFold

# import self defined functions
from riemannian_model import RiemannianModel, QuantizedRiemannianModel
from get_data import get_data

__author__ = "Michael Hersche, Tino Rellstab, Lukas Cavigelli"
__email__ = "herschmi@ethz.ch, tinor@ethz.ch, lukas.cavigelli@huawei.com"

DATA_PATH = "dataset/"
QUANTIZED = True
RIEM_OPT = "Riemann"
BANDS = [2] # [2, 4, 8, 16, 32] # 
CROSS_VALIDATION = False
CV_NO_SPLITS = 5
NO_SUBJECTS = 9
RANDOM_SEED = 1 # None
RHO = 1


def main():
    """ main function """

    if QUANTIZED:
        model = QuantizedRiemannianModel(bands=BANDS, random_state=RANDOM_SEED, riem_opt=RIEM_OPT, rho=RHO)
    else:
        model = RiemannianModel(bands=BANDS, random_state=RANDOM_SEED, riem_opt=RIEM_OPT, rho=RHO)


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

def main_export(subject, sample_idx, foldername):
    """ train the model for one pecific subject and generate pickled files """
    # we must use the quantized model
    model = QuantizedRiemannianModel(bands=BANDS, random_state=RANDOM_SEED, riem_opt=RIEM_OPT)

    train_samples, train_labels = get_data(subject, True, DATA_PATH)
    test_samples, test_labels = get_data(subject, False, DATA_PATH)

    # fit the data
    model.fit(train_samples, train_labels)

    # predict the requested sample
    test_sample = test_samples[sample_idx]
    test_label = test_labels[sample_idx]

    history = model.predict_with_intermediate(test_sample)
    history["expectation"] = test_label

    # generate the folder if it does not yet exist
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    # store everything to a file
    with open(os.path.join(foldername, "model.pkl"), "wb") as _f:
        pickle.dump(model.get_data_dict(), _f)

    # store the history
    with open(os.path.join(foldername, "verification.pkl"), "wb") as _f:
        pickle.dump(history, _f)

    # store the input only
    with open(os.path.join(foldername, "input.pkl"), "wb") as _f:
        pickle.dump(history["input"], _f)

def main_dataset_export(subject=-1, foldername='./export'):

    # we must use the quantized model
    model = QuantizedRiemannianModel(bands=BANDS, random_state=RANDOM_SEED, riem_opt=RIEM_OPT)

    if subject == -1:
        subjects = [i+1 for i in range(NO_SUBJECTS)]
    else:
        subjects = [subject]


    dataset = {'patient': subjects, 'dataset_train': list(), 'dataset_test': list()}
    for patient in subjects:

        print(f'export patient: {patient}')

        train_samples, train_labels = get_data(patient, True, DATA_PATH)
        test_samples, test_labels = get_data(patient, False, DATA_PATH)

        # fit the data
        model.fit(train_samples, train_labels)

        # predict the requested sample
        def gen_patient_dataset(samples, labels):
            dataset_patient_test = {'features_quant': list(), 'label': list()}
            for sample_idx, (sample, label) in tqdm(
                enumerate(zip(samples, labels)), 
                desc="forward pass samples", total=len(samples), leave=False):

                # gather all data
                history = model.predict_with_intermediate(sample, verbose=False)
                dataset_patient_test['features_quant'].append(history["features_quant"])
                dataset_patient_test['label'].append(label)
            return dataset_patient_test

        dataset_patient_train = gen_patient_dataset(train_samples, train_labels)
        dataset['dataset_train'].append(dataset_patient_train)
        dataset_patient_test = gen_patient_dataset(test_samples, test_labels)
        dataset['dataset_test'].append(dataset_patient_test)

    # save full dataset
    os.makedirs(foldername, exist_ok=True)
    with open(os.path.join(foldername, "dataset_full.pkl"), "wb") as _f:
        pickle.dump(dataset, _f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--export", action="store_true", help="Export the model and data")
    parser.add_argument("-d", "--dataset-export", action="store_true", help="Export full dataset with extracted features")
    parser.add_argument("-s", "--subject", type=int, default=1, help="Subject to store; specify -1 for all for dataset export")
    parser.add_argument("-i", "--sample_idx", type=int, default=0, help="Sample idx to store")
    parser.add_argument("-f", "--folder", type=str, default="export", help="Foldername to export")

    args = parser.parse_args()

    if args.export:
        main_export(args.subject, args.sample_idx, args.folder)
    elif args.dataset_export:
        main_dataset_export(args.subject, args.folder)
    else:
        main()
