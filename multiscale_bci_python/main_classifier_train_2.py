#%% import all
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import append
from tqdm import tqdm
import sklearn.svm as svm
import sklearn.tree as tree
import sklearn.neighbors as neighbors
import sklearn.neural_network as nn
import time
import random
import itertools
import functools

np.random.seed(1234)
random.seed(12345)

#%% load data
with open("./export/dataset_full.pkl", "rb") as _f:
    dataset = pickle.load(_f)

def get_patient_data(dataset, patient_idx):
    patient = dataset['patient'][patient_idx]
    dataset_train = dataset['dataset_train'][patient_idx]
    dataset_test = dataset['dataset_test'][patient_idx]

    data_train = dataset_train['features_quant']
    labels_train = dataset_train['label']
    data_test = dataset_test['features_quant']
    labels_test = dataset_test['label']
    return patient,data_train,labels_train,data_test,labels_test



# %% random sampling hyper param search
num_samples = 100

all_modes = [
    '001-baseline', 
    '010-sgdAdaptiveLR', '011-actSigmoid', 
    '012-reducedLR', '013-twoHiddenLayers', '014-oneHiddenLayer', 
    '900-linearSVM', '901-rbfSVM']

def eval_random(i, mode='001-baseline', num_samples=num_samples):

    # mode = '001-baseline'#'901-rbfSVM'#'900-linearSVM'#

    return_none = ([0.0]*9, 'skipped run', 'skipped run')

    # n1 = np.random.choice([4,8,16,32,64,128])
    # n2 = np.random.choice([4,8,16,32,64,128])
    # n3 = np.random.choice([4,8,16,32,64,128])
    # hidden_sizes = (n1, n2, n3)
    # default hparam selector
    hidden_sizes_all = list(itertools.product([4,8,16,32,64,128], [4,8,16,32,64,128], [4,8,16,32,64,128]))
    if num_samples >= len(hidden_sizes_all):
        if i < len(hidden_sizes_all):
            hidden_sizes = hidden_sizes_all[i]
        else:
            return return_none
    else: 
        hidden_sizes = random.choice(hidden_sizes_all)

    if mode == '001-baseline':
        # BASELINE
        clf = nn.MLPClassifier(
            solver='adam', learning_rate='adaptive', 
            hidden_layer_sizes=hidden_sizes, 
            random_state=1)

    elif mode == '010-sgdAdaptiveLR':
        # SGD + adaptive LR? -> No.
        clf = nn.MLPClassifier(
            solver='sgd', learning_rate='adaptive', 
            hidden_layer_sizes=hidden_sizes, 
            random_state=1)

    elif mode == '011-actSigmoid':
        # Sigmoid instead of ReLU? -> no impact -> ReLU
        clf = nn.MLPClassifier(
            solver='adam', activation='logistic', 
            hidden_layer_sizes=hidden_sizes, 
            random_state=1)

    elif mode == '012-reducedLR':
        # Reduced LR?
        clf = nn.MLPClassifier(
            solver='adam', learning_rate='adaptive', 
            learning_rate_init=0.0001,
            hidden_layer_sizes=hidden_sizes, 
            random_state=1)

    elif mode == '013-twoHiddenLayers':
        # Two hidden layers?
        hidden_sizes_all = list(itertools.product([4,8,16,32,64,128], [4,8,16,32,64,128]))
        if num_samples >= len(hidden_sizes_all):
            if i < len(hidden_sizes_all):
                hidden_sizes = hidden_sizes_all[i]
            else:
                return return_none
        else: 
            hidden_sizes = random.choice(hidden_sizes_all)
        
        clf = nn.MLPClassifier(
            solver='adam', 
            learning_rate_init=0.0001,
            hidden_layer_sizes=hidden_sizes, 
            random_state=1)

    elif mode == '014-oneHiddenLayer':
        # One hidden layer?
        hidden_sizes_all = list(itertools.product([2,4,8,16,32,64,128,256,512,1024,2048]))
        if num_samples >= len(hidden_sizes_all):
            if i < len(hidden_sizes_all):
                hidden_sizes = hidden_sizes_all[i]
            else:
                return return_none
        else: 
            hidden_sizes = random.choice(hidden_sizes_all)

        clf = nn.MLPClassifier(
            solver='adam', 
            learning_rate_init=0.001,
            hidden_layer_sizes=hidden_sizes, 
            random_state=1)

    elif mode == '900-linearSVM':
        # Linear SVM
        svm_c_all = list(itertools.product([1e-5,1e-4,1e-3,1e-2,0.1,1.0,10.0,100.0]))
        if num_samples >= len(svm_c_all):
            if i < len(svm_c_all):
                svm_c = svm_c_all[i]
            else:
                return return_none
        else: 
            svm_c = random.choice(svm_c_all)

        # clf = svm.SVC(C=svm_c[0], kernel='linear', random_state=1, tol=0.00001)
        clf = svm.LinearSVC(C=svm_c[0], loss='hinge', random_state=1, tol=0.00001)

    elif mode == '901-rbfSVM':
        # RBF SVM
        svm_c_all = list(itertools.product([1e-5,1e-4,1e-3,1e-2,0.1,1.0,10.0,100.0]))
        if num_samples >= len(svm_c_all):
            if i < len(svm_c_all):
                svm_c = svm_c_all[i]
            else:
                return return_none
        else: 
            svm_c = random.choice(svm_c_all)

        clf = svm.SVC(C=svm_c[0], kernel='rbf', random_state=1, tol=0.00001)

    else: 
        assert(False) # a valid mode is required


    if type(clf) == nn.MLPClassifier:
        exp_desc = f"num_samples: {num_samples} -- \nsolver: {clf.solver}, lr: {clf.learning_rate} (init: {clf.learning_rate_init}), act: {clf.activation}"
        run_desc = f"{clf.hidden_layer_sizes}"
    elif type(clf) == svm.SVC:
        exp_desc = f"num_samples: {num_samples} -- \n svm -- c: {clf.C}; gamma: {clf.gamma}; kernel: {clf.kernel}; deg: {clf.degree}"
        run_desc = f"{clf.C}"
    elif type(clf) == svm.LinearSVC:
        exp_desc = f"num_samples: {num_samples} -- \n svm -- c: {clf.C}; loss: {clf.loss}; tol: {clf.tol}"
        run_desc = f"{clf.C}"
    elif type(clf) == neighbors.KNeighborsClassifier:
        exp_desc = f"num_samples: {num_samples} -- \n k-NN"
        run_desc = f"{clf.n_neighbors}"
    else:
        assert(False)

    # # kNN
    # n_neighbors = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)

    # exp_desc = f"patient: {patient}, num_samples: {num_samples} -- \n k-NN"
    # run_desc = f"{n_neighbors}"

    # decision tree
    # clf = tree.DecisionTreeClassifier(criterion='entropy')

    # exp_desc = f"patient: {patient}, num_samples: {num_samples} -- \n decision tree"
    # run_desc = f"n/a"


    patient_scores = list()
    for patient_idx in range(9): #for all patients
        patient, data_train, labels_train, data_test, labels_test = get_patient_data(dataset, patient_idx)
        clf.fit(data_train, labels_train)
        score = clf.score(data_test, labels_test)
        patient_scores.append(score)

    return patient_scores, run_desc, exp_desc

data = [eval_random(i, mode='900-linearSVM') for i in tqdm(range(num_samples), desc='hparams', total=num_samples)]
scores, run_descs, exp_descs = zip(*data)


#%% get 'best' run based on patient-average accuracy
avg_scores = np.array(scores).mean(axis=1)
best_run_idx = avg_scores.argmax()
output = list()
output.append('--- best model with shared hparams across patients ---')
output.append('exp. params: ' + exp_descs[0])
output.append('hparams: ' + run_descs[best_run_idx])
for i, s in enumerate(scores[best_run_idx]):
    output.append(f"Patient {i+1}: {s}")
output.append(f"Avg. Acc.: {avg_scores[best_run_idx]}")
output.append(f"Std. dev.: {np.array(scores[best_run_idx]).std()}")
output = '\n'.join(output)
print(output)
with open(f'./export/results-patientAvgAcc-{time.strftime("%Y%m%d-%H%M%S")}.txt', 'w') as f:
    f.write(output)

#%% get 'best' run based on patient-specific hparam tuning
output = list()
output.append('--- best model with per-patient tuned hparams ---')
output.append('exp. params: ' + exp_descs[0].replace('\n', ' '))
best_scores = np.array(scores).max(axis=0)
best_scores_runidx = np.array(scores).argmax(axis=0)
for i, (s, rdi) in enumerate(zip(scores[best_run_idx], best_scores_runidx)): # BUG: best_run_idx not set?!
    output.append(f"Patient {i+1}: {s} -- {run_descs[rdi]}")
output.append(f"Avg. Acc.: {best_scores.mean()}")
output.append(f"Std. dev.: {best_scores.std()}")
output = '\n'.join(output)
print(output)
with open(f'./export/results-patientBestAcc-{time.strftime("%Y%m%d-%H%M%S")}.txt', 'w') as f:
    f.write(output)


#%% multiple classifier runs (execute)
num_samples = 1000
data_all = [[
    eval_random(i, mode=mode, num_samples=num_samples) 
    for i in tqdm(range(num_samples), desc='hparams', total=num_samples)] 
    for mode in all_modes]
with open(f'./export/results-multiClassifier-numSamples_{num_samples}-{time.strftime("%Y%m%d-%H%M%S")}.pkl', 'wb') as f:
    pickle.dump((data_all, num_samples, all_modes), f)

#%% analysis of multiple classifier runs (analyze)
# fname = './export/results-multiClassifier-numSamples_2-20210805-022148.pkl'
fname = './export/results-multiClassifier-numSamples_1000-20210805-043926.pkl'
with open(fname, 'rb') as f:
    data_all, num_samples, all_modes = pickle.load(f)

# all data
# data_all_fused = [
#     (patient_scores, f"mode: [{mode}], exp: [{exp_desc}], run: [{run_desc}]") 
#     for patient_scores, run_desc, exp_desc in data 
#     for data, mode in zip(data_all, all_modes)]
#or for specific data only
sel_modes = all_modes
# sel_modes = ['900-linearSVM']
# sel_modes = ['001-baseline']
data_all_fused = [
    (patient_scores, f"mode: [{sel_mode}], exp: [{exp_desc}], run: [{run_desc}]") 
    for sel_mode in sel_modes
    for patient_scores, run_desc, exp_desc in data_all[all_modes.index(sel_mode)]
   ]
scores, descs = zip(*data_all_fused)

# best run based on patient-average accuracy
avg_scores = np.array(scores).mean(axis=1)
best_run_idx = avg_scores.argmax()
output = list()
output.append('--- best model with shared hparams across patients ---')
output.append('desc: ' + descs[best_run_idx])
for i, s in enumerate(scores[best_run_idx]):
    output.append(f"Patient {i+1}: {s*100:.2f}%")
output.append(f"Avg. Acc.: {avg_scores[best_run_idx]*100:.2f}%")
output.append(f"Std. dev.: {np.array(scores[best_run_idx]).std()*100:.2f}%")
output = '\n'.join(output)
print(output)

# best run (accuracy) for patient-specific classifier & hparams
output = list()
output.append('--- best model with per-patient tuned hparams ---')
best_scores = np.array(scores).max(axis=0)
best_scores_runidx = np.array(scores).argmax(axis=0)
for i, rdi in enumerate(best_scores_runidx):
    output.append(f"Patient {i+1}: {scores[rdi][i]*100:.2f}%")
    # output.append(f"    -- {descs[rdi]}")
output.append(f"Avg. Acc.: {best_scores.mean()*100:.2f}%")
output.append(f"Std. dev.: {best_scores.std()*100:.2f}%")
output = '\n'.join(output)
print(output)



#%% plot hparam search results
# data_unique = dict()
# for acc, hparam in list(set(zip(scores, run_descs))):
#     if not(hparam in data_unique.keys() and data_unique[hparam] >= acc):
#         data_unique[hparam] = acc
# data_unique_sorted = sorted([(v, k) for k, v in data_unique.items()], key=lambda d: d[0], reverse=True)
# scores, run_descs = zip(*data_unique_sorted)
# exp_desc = exp_descs[0] + f' -- \nmax. score: {scores[0]}'

# num_show = 10
# plt.bar(run_descs[:num_show], scores[:num_show])
# plt.xticks(rotation=90)
# plt.grid(axis='y')
# plt.ylim(0.5, 1.0)
# plt.title(exp_desc)
# timestr = time.strftime("%Y%m%d-%H%M%S")
# plt.savefig(f'save_fig-{timestr}.pdf', bbox_inches='tight')


# # %%
# TODO: what to plot???!!!
# 1) updated table for various methods
# 2) best hyper parameter variant
# 3) best hparam + method variant
# 4) trade-off parameter count vs. avg. accuracy
# 5) TURN OFF THE logm quantization!
    #TODO: ideas -- 1) mixing like in EEG-TCnet, 2) try MLP + hinge loss, 3) augmentation?

# %%
