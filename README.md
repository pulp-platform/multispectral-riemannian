# multiscale-bci-wolf

Implementation of the quantized Multiscale Riemannian Classifier on Mr Wolf

# Setup

First, get all the submodules

```
git submodule update --init --recursive
```

Then, make sure to setup the virtual environment properly

```
python -m virtualenv multiscale_bci_python/env
source multiscale_bci_python/env/bin/activate
pip install numpy scipy scikit-learn pyriemann matplotlib tqdm
```
