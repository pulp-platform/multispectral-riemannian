Copyright (C) 2020 ETH Zurich, Switzerland. Refer to paragraph `License and Attribution` below for information on the license.

# multiscale-bci-wolf

Implementation of the quantized Multiscale Riemannian Classifier on Mr Wolf.

This project was developed by Tibor Schneider during his Semester thesis at ETH Zurich under the supervision of Xiaying Wang, Michael Hersche, and Lukas Cavigelli.

For more details, please read the paper *Mixed-Precision Quantization and Parallel Implementation of Multispectral Riemannian Classification for Brain--Machine Interfaces* available on [IEEE Xplore](https://ieeexplore.ieee.org/document/9401564) and on [arXiv](https://arxiv.org/abs/2102.11221). If you find this work useful in your research, please cite
```
@INPROCEEDINGS{9401564,
author={Wang, Xiaying and Schneider, Tibor and Hersche, Michael and Cavigelli, Lukas and Benini, Luca},
booktitle={2021 IEEE International Symposium on Circuits and Systems (ISCAS)},
title={Mixed-Precision Quantization and Parallel Implementation of Multispectral Riemannian Classification for Brain--Machine Interfaces},
year={2021},
pages={1-5},
doi={10.1109/ISCAS51556.2021.9401564}
}
```

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

First, compile the shared library for the python model:

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

Add the option `b` to execute the code on the Board. The parameter `f` keeps the compiler from using the FMA unit. Also, the option `d` forces the compiler to use soft division and square root. See the help page displayed with `./run.sh -h`.

# License and Attribution

This repository is released under the Apache-2.0 license, see LICENSE file for details, except for the codes in the folder `multiscale_bci_python` which is developed based on [MultiScale-BCI/IV-2a](https://github.com/MultiScale-BCI/IV-2a) repository, released under MIT License. Refer to the original repository for more details.
