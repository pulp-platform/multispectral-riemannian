#!/bin/bash

make clean all run PMSIS_OS=pulp-os platform=gvsoc

ar rcs dsp/libplpdsp.a BUILD/GAP9/GCC_RISCV/dsp/*.o BUILD/GAP9/GCC_RISCV/dsp/kernels/*.o
