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
pip install numpy scipy sklearn pyriemann matplotlib tqdm
```

Also, make sure to get the BCI Competition IV dataset 2a and store it in `multiscale_bci_python/dataset`
