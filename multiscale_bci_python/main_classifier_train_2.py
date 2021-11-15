#%% import all
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import append
from tqdm import tqdm
import sklearn
import sklearn.pipeline
import sklearn.preprocessing
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
fname_dataset = "./export/dataset_full_v2_full.pkl"
fname_dataset = "./export/dataset_full_v2_quant.pkl"
with open(fname_dataset, "rb") as _f:
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
    '900-linearSVM', '901-rbfSVM', '902-scaledLinearSVM']
all_modes = [
    '001-baseline', 
    '011-actSigmoid', 
    '014-oneHiddenLayer', 
    '900-linearSVM', '902-scaledLinearSVM'] # '013-twoHiddenLayers', '012-reducedLR', '010-sgdAdaptiveLR', '901-rbfSVM'

def eval_random(i, mode='001-baseline', num_samples=num_samples, patient_idx_list=None):

    # mode = '001-baseline'#'901-rbfSVM'#'900-linearSVM'#

    if patient_idx_list == None:
        patient_idx_list = list(range(9))

    return_none = ([0.0]*len(patient_idx_list), 'skipped run', 'skipped run', None)

    # n1 = np.random.choice([4,8,16,32,64,128])
    # n2 = np.random.choice([4,8,16,32,64,128])
    # n3 = np.random.choice([4,8,16,32,64,128])
    # hidden_sizes = (n1, n2, n3)
    # default hparam selector
    hidden_sizes_all = list(itertools.product([4,8,16,32,64,128,256], [4,8,16,32,64,128,256], [4,8,16,32,64,128,256]))
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
        hidden_sizes_all = list(itertools.product([4,8,16,32,64,128,256], [4,8,16,32,64,128,256]))
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
        # svm_c_all = list(itertools.product([1e-5,1e-4,1e-3,1e-2,0.1,1.0,10.0,100.0]))
        svm_c_all = list(itertools.product([0.1]))
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

    elif mode == '902-scaledLinearSVM':
        # Linear SVM
        # svm_c_all = list(itertools.product([1e-5,1e-4,1e-3,1e-2,0.1,1.0,10.0,100.0]))
        svm_c_all = list(itertools.product([0.1]))
        if num_samples >= len(svm_c_all):
            if i < len(svm_c_all):
                svm_c = svm_c_all[i]
            else:
                return return_none
        else: 
            svm_c = random.choice(svm_c_all)

        clf = sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.MinMaxScaler(),
            svm.LinearSVC(C=svm_c[0], loss='hinge', random_state=1, tol=0.00001)
            )

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
    elif mode == '902-scaledLinearSVM':
        exp_desc = f"num_samples: {num_samples} -- \n svm -- c: {clf[-1].C}; loss: {clf[-1].loss}; tol: {clf[-1].tol}"
        run_desc = f"{clf[-1].C}"
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
    patient_models = list()
    for patient_idx in patient_idx_list: #for all patients
        patient, data_train, labels_train, data_test, labels_test = get_patient_data(dataset, patient_idx)
        clf.fit(data_train, labels_train)
        score = clf.score(data_test, labels_test)
        patient_scores.append(score)
        patient_models.append(clf)

    # return_models = True
    # if return_models:
    return patient_scores, run_desc, exp_desc, patient_models
    # else:
        # return patient_scores, run_desc, exp_desc

data = [eval_random(i, mode='900-linearSVM') for i in tqdm(range(num_samples), desc='hparams', total=num_samples)]
scores, run_descs, exp_descs, models = zip(*data)


# #%% get 'best' run based on patient-average accuracy
# avg_scores = np.array(scores).mean(axis=1)
# best_run_idx = avg_scores.argmax()
# output = list()
# output.append('--- best model with shared hparams across patients ---')
# output.append('exp. params: ' + exp_descs[0])
# output.append('hparams: ' + run_descs[best_run_idx])
# for i, s in enumerate(scores[best_run_idx]):
#     output.append(f"Patient {i+1}: {s}")
# output.append(f"Avg. Acc.: {avg_scores[best_run_idx]}")
# output.append(f"Std. dev.: {np.array(scores[best_run_idx]).std()}")
# output = '\n'.join(output)
# print(output)
# with open(f'./export/results-patientAvgAcc-{time.strftime("%Y%m%d-%H%M%S")}.txt', 'w') as f:
#     f.write(output)

# #%% get 'best' run based on patient-specific hparam tuning
# output = list()
# output.append('--- best model with per-patient tuned hparams ---')
# output.append('exp. params: ' + exp_descs[0].replace('\n', ' '))
# best_scores = np.array(scores).max(axis=0)
# best_scores_runidx = np.array(scores).argmax(axis=0)
# for i, (s, rdi) in enumerate(zip(scores[best_run_idx], best_scores_runidx)): # BUG: best_run_idx not set?!
#     output.append(f"Patient {i+1}: {s} -- {run_descs[rdi]}")
# output.append(f"Avg. Acc.: {best_scores.mean()}")
# output.append(f"Std. dev.: {best_scores.std()}")
# output = '\n'.join(output)
# print(output)
# with open(f'./export/results-patientBestAcc-{time.strftime("%Y%m%d-%H%M%S")}.txt', 'w') as f:
#     f.write(output)


#%% multiple classifier runs (execute)
num_samples = 1000
patient_idx_list = list(range(9))
patient_idx_list = [3, 7]

data_all = [[
    eval_random(
        i, mode=mode, 
        num_samples=num_samples, 
        patient_idx_list=patient_idx_list) 
    for i in tqdm(range(num_samples), desc='hparams', total=num_samples)] 
    for mode in all_modes]
fname = f'./export/results-multiClassifier-numSamples_{num_samples}-{time.strftime("%Y%m%d-%H%M%S")}.pkl'
with open(fname, 'wb') as f:
    pickle.dump((data_all, num_samples, all_modes), f)

#%% analysis of multiple classifier runs (analyze)
# fname = './export/results-multiClassifier-numSamples_1000-20210825-051413.pkl'
# fname = './export/results-multiClassifier-numSamples_1000-20210825-142442.pkl'
# fname = './export/results-multiClassifier-numSamples_1000-20210825-153024.pkl'
# fname = './export/results-multiClassifier-numSamples_1000-20210826-014226.pkl'
# fname = './export/results-multiClassifier-numSamples_1000-20210826-022023.pkl'
# fname = './export/results-multiClassifier-numSamples_1000-20210826-044702.pkl' # full precision
fname = './export/results-multiClassifier-numSamples_1000-20210826-044755.pkl' # quantized
with open(fname, 'rb') as f:
    file_complete = pickle.load(f)
# data_all, num_samples, all_modes, all_models = file_complete
data_all, num_samples, all_modes = file_complete

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
    (patient_scores, f"mode: [{sel_mode}], exp: [{exp_desc}], run: [{run_desc}]", patient_models) 
    for sel_mode in sel_modes
    for patient_scores, run_desc, exp_desc, patient_models in data_all[all_modes.index(sel_mode)]
    # for patient_scores, run_desc, exp_desc in data_all[all_modes.index(sel_mode)]
   ]
scores, descs, models = zip(*data_all_fused)

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
output.append('--- best model with per-patient tuned hparams and classifier ---')
best_scores = np.array(scores).max(axis=0)
best_scores_runidx = np.array(scores).argmax(axis=0)
selected_models = [models[rdi][i] for i, rdi in enumerate(best_scores_runidx)]
for i, rdi in enumerate(best_scores_runidx):
    output.append(f"Patient {i+1}: {scores[rdi][i]*100:.2f}%")
    output.append(f" -- {descs[rdi]}")
    m = selected_models[i]
    if type(m) == svm.LinearSVC:
        nparams = m.coef_.size
    elif type(m) == nn.MLPClassifier:
        nparams = np.array([c.size for c in m.coefs_]).sum()
    else:
        nparams = "unknown classifier"
    output.append(f" -- model: {m}")
    output.append(f" -- params: {nparams}")
output.append(f"Avg. Acc.: {best_scores.mean()*100:.2f}%")
output.append(f"Std. dev.: {best_scores.std()*100:.2f}%")
output = '\n'.join(output)
print(output)
with open(f'./export/results-patientBestAcc-TEMP2.txt', 'w') as f:
    f.write(output)



#%% accuracy vs. #params plot

def get_params(m):
    if m == None:
        nparams = 1e12
    elif type(m) == svm.LinearSVC:
        nparams = m.coef_.size
    elif type(m) == nn.MLPClassifier:
        nparams = np.array([c.size for c in m.coefs_]).sum()
    else:
        nparams = -1#"unknown classifier"
    return nparams
def fix_none(sm):
    if sm == None:
        return [None]*9
    else: 
        return sm
scores_all = np.array(scores) # method x subject
params_all = np.array([[get_params(m) for m in fix_none(subj_models)] for subj_models in models])

params_unique = np.unique(params_all)
params_unique = params_unique[params_unique <= 2e6]
def get_metric_by_paramConstr(max_params):
    constr_scores = scores_all * (params_all <= max_params)
    constr_scores_best = constr_scores.max(axis=0)
    accuracy_avg = constr_scores_best.mean()
    accuracy_std = constr_scores_best.std()
    return accuracy_avg, accuracy_std

acc_avg, acc_std = zip(*[get_metric_by_paramConstr(max_params) for max_params in params_unique])

plt.plot(params_unique[:-1], np.array(acc_avg[:-1])*100)
plt.xlabel('#params')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.show()

#%% accuracy vs. #params plot -- COMPLETE


maccs = [130,130, 630, 1040, 2020, 68, 0]#, 964, 17300] # in 0.1M

parameters = [2548/4,2548, 47300, 261000, 155000, 4270, 4554/2]#, 240000,#7780000] in 8 bits B

fmaps = [65.602/4, 225, 1013, 50, 5775, 396, 38.544/4+4.554+4.554]#, 499, 525] # in 8 bits kB

kb32 = [x * 4/1000 for x in parameters]
fmaps32 = [x * 4 for x in fmaps]
memfoot = [x+y for x, y in zip(kb32, fmaps32)]

accuracies = [70.9, 71.2, 74.31, 73.70, 75.80, 77.34, 74.1]#, 81.1, 88.87]

networks = ['Q-EEGNet','EEGNet', 'S.ConvNet', 'FBCSP', 'MSFBCNN', 'EEG-TCNet', 'MRC-Mr.Wolf']#, 'CNN++', 'TPCT']

#new = [x/1000000 for x in maccs]
#newlist = [x/100000 for x in maccs]
textObjs = []
avoidObjs = []
fig = plt.figure(num=None, figsize=(8, 6), dpi=800, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)

for numParam,acc1,ar,name in zip(memfoot, accuracies,maccs,networks):
    x = numParam
    y = acc1
    s = ar
    if(name == 'Variable EEG-TCNet'):
        txtobj = ax.text(x, 84, name, weight='bold',fontsize = 10, horizontalalignment='center')
        sctobj = ax.scatter(x, y, s=ar,c='#32FF32',hatch='////')
        ax.scatter(x,y,c='k',s=2)
    elif(name == 'EEG-TCNet'):
        txtobj = ax.text(x, y-1, name, fontsize = 10, horizontalalignment='center') #weight='bold',
        sctobj = ax.scatter(x, y, s=ar,c='tab:red')
        ax.scatter(x,y,c='silver',s=2)
    elif(name == 'Variable EEGNet'):
        txtobj = ax.text(x-4400, 81, name, fontsize = 10)
        sctobj = ax.scatter(x, y, s=ar,c='#000096',ec='white',hatch='////')
        ax.scatter(x,y,c='k',s=2)
    elif(name == 'EEGNet'):
        txtobj = ax.text(x, y-1, name, fontsize = 10,horizontalalignment='center')
        sctobj = ax.scatter(x, y, s=ar,c='tab:red')
        ax.scatter(x,y,c='silver',s=2)
    elif(name == 'Q-EEGNet'):
        txtobj = ax.text(x, y-1, name, fontsize = 10,horizontalalignment='center')
        sctobj = ax.scatter(x, y, s=ar,c='tab:red')
        ax.scatter(x,y,c='silver',s=2)
    elif(name == 'TPCT'):
        txtobj = ax.text(x, 80.2, name,horizontalalignment='center')
        sctobj = ax.scatter(x, y, s=ar,c='#800000')
        ax.scatter(x,y,c='silver',s=2)
    elif(name == 'MSFBCNN'):
        txtobj = ax.text(x, y-1, name,horizontalalignment='center')
        sctobj = ax.scatter(x, y, s=ar,c='tab:red')
        ax.scatter(x,y,c='silver',s=2)
    elif(name == 'CNN++'):
        txtobj = ax.text(x, 83, name,horizontalalignment='center')
        sctobj = ax.scatter(x, y, s=ar,c='#BA55D3')
        ax.scatter(x,y,c='silver',s=2)
    elif(name == 'DFFN'):
        txtobj = ax.text(x, 82, name,horizontalalignment='center')
        sctobj = ax.scatter(x, y, s=ar,ec='white',hatch='////')
        ax.scatter(x,y,c='silver',s=2)
    elif(name == 'S.ConvNet'):
        txtobj = ax.text(x, y-1, name,horizontalalignment='center')
        sctobj = ax.scatter(x, y, s=ar, c='tab:red')
        ax.scatter(x,y,c='silver',s=2)
    elif(name == 'MRC-Mr.Wolf'):
        txtobj = ax.text(x, y-1, name,horizontalalignment='center', c='#008f1f')
        sctobj = ax.scatter(x, y, s=1000000/10000,marker = 's', c='#008f1f') #c='tab:red')
        ax.scatter(x,y,c='silver',s=2)
    elif(name == 'FBCSP'):
        txtobj = ax.text(x, y-1, name,horizontalalignment='center')
        sctobj = ax.scatter(x, y, s=ar, c='tab:red')
        ax.scatter(x,y,c='silver',s=7)
#     elif(name == 'FB-3D-CNN'):
#         txtobj = ax.text(x, y-1, name,horizontalalignment='center')
#         sctobj = ax.scatter(x, y, s=ar, c='tab:green')
#         ax.scatter(x,y,c='silver',s=2)
    textObjs.append(txtobj)
    avoidObjs.append(sctobj)
    
xmax = 50*10**4
xmin = 10
ymax = 80
ymin = 64
L1mem = 64
L2mem = 512
L1memvega = 128
L2memvega = 1500

plt.vlines(L1mem, ymin,ymax, colors='grey', ls='dashed')
#ax.text(L1mem+L1mem*0.1, ymin+0.5, '64kB')
ax.text(L1mem, ymax+0.2, '64kB', fontsize = 10, horizontalalignment='center')
plt.vlines(L2mem, ymin,ymax, colors='grey', ls='dashed')
#ax.text(L2mem+L2mem*0.1, ymin+0.5, '512kB')
ax.text(L2mem, ymax+0.2, '512kB', fontsize = 10, horizontalalignment='center')

plt.vlines(L1memvega, ymin,ymax, colors='grey', ls='-.')
#ax.text(L1mem+L1mem*0.1, ymin+0.5, '64kB')
ax.text(L1memvega, ymax+0.2, '128kB', fontsize = 10, horizontalalignment='center')
plt.vlines(L2memvega, ymin,ymax, colors='grey', ls='-.')
#ax.text(L2mem+L2mem*0.1, ymin+0.5, '512kB')
ax.text(L2memvega, ymax+0.2, '1500kB', fontsize = 10, horizontalalignment='center')

ax.set_xscale('log')
ax.set_xlabel('Memory footprint [kB]', fontsize=14)
ax.set_ylabel('Accuracy [%]', fontsize=14)
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
plt.grid(True,ls='--')
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)

#Show blob sizes:
x_blobs = [10**4,2*10**4,5*10**4,2*10**5]#,1.3*10**7]
y_blobs = [66 for i in x_blobs]
y_blobs_text = [66.5,66.8,67.2,67.8]#,72]
s_blobs = [100,200,500,2000]#,10000]
names_blobs = ['10M','20M','50M','200M']#,'1000M']
ax.scatter(x_blobs,y_blobs,s=s_blobs,c='0.9')
for x_b,y_b,n in zip(x_blobs,y_blobs_text,names_blobs):
    ax.text(x_b,y_b,n, fontsize=9,horizontalalignment='center',style='italic')
#adjust_text(textObjs, add_objects=avoidObjs#, 
            #arrowprops=dict(arrowstyle="->", color='black')
#            )

# plt.plot(-1, -1, marker='o', linestyle="None", mfc='tab:green', markeredgecolor='tab:green', label='2-class')
# plt.plot(-1, -1, marker='o', linestyle="None", mfc='tab:blue', markeredgecolor='tab:blue', label='3-class')
# plt.plot(-1, -1, marker='o', linestyle="None", mfc='tab:red', markeredgecolor='tab:red', label='4-class')
# plt.legend(loc='lower center', ncol=3)

memory_req = np.array(params_unique[:-1]) + (2*22*876+18*(22+1)*22/2*4+18*(22+1)*22/2*2)
accuracy = np.array(acc_avg[:-1])
# memory_req = np.insert(memory_req, 0, 123)
plt.plot(memory_req/1000, accuracy*100)
ax.text(280, 75, 'MRC Vega', horizontalalignment='center', color='#1f77b4')


plt.tight_layout()
#plt.savefig('figures/accuracy_vs_parameters_without_arrows_without_colorbar.eps', format='eps')
#plt.savefig('figures/see_text.eps', format='eps')
# plt.savefig('./acc_vs_memory.svg', format='svg')
plt.savefig('./export/acc_vs_memory.pdf')
#tikzplotlib.save("acc_vs_memory_wo_cs.tex")
plt.show()



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


#TODO: ideas -- 1) mixing like in EEG-TCnet, 2) try MLP + hinge loss, 3) augmentation?

# %%
