# Fast and Accurate Multiclass Inference for MI-BCIs Using Large Multiscale Temporal and Spectral Features

This is the code of an accepted conference paper submitted to EUSIPCO 2018. The preprint is available on this arXiv [link](https://arxiv.org/abs/1806.06823). If you are using this code please cite our paper. 

## Getting Started

First, download the source code.
Then, download the dataset "Four class motor imagery (001-2014)" of the [BCI competition IV-2a](http://bnci-horizon-2020.eu/database/data-sets). Put all files of the dataset (A01T.mat-A09E.mat) into a subfolder within the project called 'dataset' or change self.data_path in main_csp and main_riemannian. 

```
mkdir dataset && cd dataset
wget http://bnci-horizon-2020.eu/database/data-sets/001-2014/A01T.mat
wget http://bnci-horizon-2020.eu/database/data-sets/001-2014/A02T.mat
wget http://bnci-horizon-2020.eu/database/data-sets/001-2014/A03T.mat
wget http://bnci-horizon-2020.eu/database/data-sets/001-2014/A04T.mat
wget http://bnci-horizon-2020.eu/database/data-sets/001-2014/A05T.mat
wget http://bnci-horizon-2020.eu/database/data-sets/001-2014/A06T.mat
wget http://bnci-horizon-2020.eu/database/data-sets/001-2014/A07T.mat
wget http://bnci-horizon-2020.eu/database/data-sets/001-2014/A08T.mat
wget http://bnci-horizon-2020.eu/database/data-sets/001-2014/A09T.mat
wget http://bnci-horizon-2020.eu/database/data-sets/001-2014/A01E.mat
wget http://bnci-horizon-2020.eu/database/data-sets/001-2014/A02E.mat
wget http://bnci-horizon-2020.eu/database/data-sets/001-2014/A03E.mat
wget http://bnci-horizon-2020.eu/database/data-sets/001-2014/A04E.mat
wget http://bnci-horizon-2020.eu/database/data-sets/001-2014/A05E.mat
wget http://bnci-horizon-2020.eu/database/data-sets/001-2014/A06E.mat
wget http://bnci-horizon-2020.eu/database/data-sets/001-2014/A07E.mat
wget http://bnci-horizon-2020.eu/database/data-sets/001-2014/A08E.mat
wget http://bnci-horizon-2020.eu/database/data-sets/001-2014/A09E.mat
```

### Prerequisites

- python3
- numpy
- sklearn
- pyriemann
- scipy

The packages can be installed easily with conda and the _config.yml file: 
```
$ conda env create -f _config.yml -n msenv
$ conda activate msenv 
```

### Recreate results

For the recreation of the CSP results run main_csp.py. 
Change self.svm_kernel for testing different kernels:
- self.svm_kernel='linear'  -> self.svm_c = 0.05
- self.svm_kernel='rbf'     -> self.svm_c = 20
- self.svm_kernel='poly'    -> self.svm_c = 0.1

```
$ python main_csp.py
```
For the recreation of the Riemannian results run main_riemannian.py. 
Change self.svm_kernel for testing different kernels:
- self.svm_kernel='linear'  -> self.svm_c = 0.1
- self.svm_kernel='rbf'     -> self.svm_c = 20

Change self.riem_opt for testing different means:
- self.riem_opt = "Riemann"
- self.riem_opt = "Riemann_Euclid" 
- self.riem_opt = "Whitened_Euclid"
- self.riem_opt = "No_Adaptation"

```
$ python main_riemannian.py
```

### Export full dataset for complex classifiers
Running 
```
python main_riemannian.py -d -s="-1"
python main_classifier_train_2.py
```
will generate the file `./export/dataset_full.pkl` of all subjects/patients and samples and both the training and test data. 

## Authors

* **Michael Hersche** - *Initial work* - [MHersche](https://github.com/MHersche)
* **Tino Rellstab** - *Initial work* - [tinorellstab](https://github.com/tinorellstab)
* **Lukas Cavigelli** - *Modifications for Vega* - [lukasc-ch](https://github.com/lukasc-ch)
