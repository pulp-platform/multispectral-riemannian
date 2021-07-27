#!/bin/bash

# set the conda base prefix, where the conda folder is installed.
CONDA_BASE_PREFIX=~/miniconda3

# source vega environment
source ~/vega_sdk/configs/gap9.sh

# build the subsets of dsp library. vega sdk is not stable yet, hence we
# manually build the functions needed in this application.

make clean all run PMSIS_OS=pulp-os platform=gvsoc

ar rcs dsp/libplpdsp.a BUILD/GAP9/GCC_RISCV/dsp/*.o BUILD/GAP9/GCC_RISCV/dsp/kernels/*.o

