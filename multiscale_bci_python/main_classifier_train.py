#%% import all
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sklearn.svm as svm
import sklearn.tree as tree
import sklearn.neighbors as neighbors
import sklearn.neural_network as nn
import time

#%% load data
foldername = './export/'
with open(os.path.join(foldername, "dataset_full.pkl"), "rb") as _f:
    dataset = pickle.load(_f)

# for patient, dataset_train, dataset_test in zip(
#     dataset['patient'], dataset['dataset_train'], dataset['dataset_test']):
patient_idx = 8
patient = dataset['patient'][patient_idx]
dataset_train = dataset['dataset_train'][patient_idx]
dataset_test = dataset['dataset_test'][patient_idx]

data_train = dataset_train['features_quant']
labels_train = dataset_train['label']
data_test = dataset_test['features_quant']
labels_test = dataset_test['label']

print(f"patient: {patient}")
print(f"num feature vectors: {len(data_train)}")
print(f"feat vector shape: {data_train[0].shape}")

# #%% Train single network
# # clf = nn.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(40, 40, 4), random_state=1)
# clf = nn.MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(40, 40, 4), random_state=1)
# clf.fit(data_train, labels_train)
# # predict_test = clf.predict(data_test)
# score = clf.score(data_test, labels_test)
# print(f"score: {score}")

# #%% hyper param search -- 2D grid
# def eval(hidden_sizes):
#     print(hidden_sizes)
#     clf = nn.MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=hidden_sizes, random_state=1)
#     clf.fit(data_train, labels_train)
#     return clf.score(data_test, labels_test)
# num1 = [2,4,8,16]#,32,64,128]
# num2 = [2,4,8,16]#,32,64,128]
# res = [[eval((n1, n2)) for n1 in num1] for n2 in num2]

# #%% plot results of 2D search
# # plt.pcolormesh(np.array(res))
# data = np.array(res)
# plt.matshow(data)
# plt.colorbar()
# for (i, j), z in np.ndenumerate(data):
#     plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
# plt.xlabel('n1')
# plt.ylabel('n2')
# plt.yticks(list(range(len(num2))), labels=num2)
# plt.xticks(list(range(len(num1))), labels=num1)
# plt.show()
# plt.savefig('subj01-2layer-relu.pdf')

# %% random sampling hyper param search
num_samples = 100

def eval_random(i):
    n1 = np.random.choice([2,4,8,16,32,64,128])
    n2 = np.random.choice([2,4,8,16,32,64,128])
    n3 = np.random.choice([2,4,8,16,32,64,128])
    hidden_sizes = (n1, n2, n3)

    # BASELINE
    clf = nn.MLPClassifier(
        solver='adam', learning_rate='adaptive', 
        hidden_layer_sizes=hidden_sizes, 
        random_state=1)

    # SGD + adaptive LR? -> No.
    # clf = nn.MLPClassifier(
    #     solver='sgd', learning_rate='adaptive', 
    #     hidden_layer_sizes=hidden_sizes, 
    #     random_state=1)

    # Sigmoid instead of ReLU? -> no impact -> ReLU
    # clf = nn.MLPClassifier(
    #     solver='adam', activation='logistic', 
    #     hidden_layer_sizes=hidden_sizes, 
    #     random_state=1)

    # Reduced LR?
    # clf = nn.MLPClassifier(
    #     solver='adam', learning_rate='adaptive', 
    #     learning_rate_init=0.0001,
    #     hidden_layer_sizes=hidden_sizes, 
    #     random_state=1)

    # Two hidden layers?
    # hidden_sizes = hidden_sizes[:2]
    # clf = nn.MLPClassifier(
    #     solver='adam', 
    #     learning_rate_init=0.001,
    #     hidden_layer_sizes=hidden_sizes, 
    #     random_state=1)

    # One hidden layer?
    # hidden_sizes = (np.random.choice([2,4,8,16,32,64,128,256,512,1024,2048]),)
    # clf = nn.MLPClassifier(
    #     solver='adam', 
    #     learning_rate_init=0.001,
    #     hidden_layer_sizes=hidden_sizes, 
    #     random_state=1)

    exp_desc = f"patient: {patient}, num_samples: {num_samples} -- \nsolver: {clf.solver}, lr: {clf.learning_rate} (init: {clf.learning_rate_init}), act: {clf.activation}"
    run_desc = f"{hidden_sizes}"

    # svm_c = np.random.choice([1e-5,1e-4,1e-3,1e-2,0.1,1.0,10.0,100.0])
    # Linear SVM?
    # clf = svm.SVC(C=svm_c, kernel='linear', random_state=1, tol=0.00001)
    # RBF SVM? -> much worse than linear SVM
    # clf = svm.SVC(C=svm_c, kernel='rbf', random_state=1, tol=0.00001)

    # exp_desc = f"patient: {patient}, num_samples: {num_samples} -- \n svm -- c: {clf.C}; gamma: {clf.gamma}; kernel: {clf.kernel}; deg: {clf.degree}"
    # run_desc = f"{svm_c}"

    # # kNN
    # n_neighbors = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)

    # exp_desc = f"patient: {patient}, num_samples: {num_samples} -- \n k-NN"
    # run_desc = f"{n_neighbors}"

    # decision tree
    # clf = tree.DecisionTreeClassifier(criterion='entropy')

    # exp_desc = f"patient: {patient}, num_samples: {num_samples} -- \n decision tree"
    # run_desc = f"n/a"

    #TODO: ideas -- 1) mixing like in EEG-TCnet, 2) try MLP + hinge loss, 3) augmentation?

    clf.fit(data_train, labels_train)
    score = clf.score(data_test, labels_test)

    return score, run_desc, exp_desc

data = [eval_random(i) for i in tqdm(range(num_samples), desc='hparams', total=num_samples)]
scores, run_descs, exp_descs = zip(*data)

#%% plot hparam search results
data_unique = dict()
for acc, hparam in list(set(zip(scores, run_descs))):
    if not(hparam in data_unique.keys() and data_unique[hparam] >= acc):
        data_unique[hparam] = acc
data_unique_sorted = sorted([(v, k) for k, v in data_unique.items()], key=lambda d: d[0], reverse=True)
scores, run_descs = zip(*data_unique_sorted)
exp_desc = exp_descs[0] + f' -- \nmax. score: {scores[0]}'

num_show = 10
plt.bar(run_descs[:num_show], scores[:num_show])
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.ylim(0.5, 1.0)
plt.title(exp_desc)
timestr = time.strftime("%Y%m%d-%H%M%S")
plt.savefig(f'save_fig-{timestr}.pdf', bbox_inches='tight')


# %%


