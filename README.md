# multiscale-bci-wolf

Implementation of the quantized Multiscale Riemannian Classifier on Mr Wolf

# Requirements

- Python 2.7 and 3.5 (for PULP-SDK)
- Python 3.8 for the multiscale riemannian python model
- [PULP-SDK](https://github.com/pulp-platform/pulp-sdk "PULP-SDK repository")
- [PULP-DSP](https://github.com/pulp-platform/pulp-dsp "PULP-DSP repository")
- The [BCI Competition IV 2a](http://www.bbci.de/competition/iv/ "BCI Competition IV") dataset (in .mat format) stored at (or linked to) `multiscale_bci_python/dataset`
- The following pip modules (for python 3.8): `numpy`, `scipy`, `sklearn`, `pyriemann`, `matplotlib`, `tqdm`, `cffi`, which can be installed by:

```
python3.8 -e pip install numpy scipy sklearn pyriemann matplotlib tqdm cffi
```

# Setup

First, get all the submodules

```
git submodule update --init --recursive
```

Then, compile the shared library for the python model:

```
cd multiscale_bci_python
make all
```

# Usage

The script `run.sh` does everything to run the program. See all possible arguments and the usage with `./run.sh -h`.

When using this for the first time, you must train the network and export the necessary data. For this, run the command (the option `t` tells the script to train the network, and the option `n` keeps it from running the code):

```
./run.sh -tn
```

Then, you can run it with :

```
./run.sh [-bfd]
```

Add the option `b` to execute the code on the Board. The parameter `f` keeps the compiler from using the FMA unit. Also, the option `d` forces the compiler to use soft division ans square root. See the help page displayed with `./run.sh -h`.
