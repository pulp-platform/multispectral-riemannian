#!/bin/bash
#
# Copyright (C) 2020 ETH Zurich. All rights reserved.
#
# Author: Tibor Schneider, ETH Zurich
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

PLATFORM="gvsoc"
TRAIN=false
export WOLFTEST_USE_FMA=true
export WOLFTEST_USE_SQRTDIV=true
export WOLFTEST_EPSILON=1e-35

while getopts "bp:tfde:ah" name; do
    case "$name" in
        b) PLATFORM="board";;
        p) PLATFORM=$OPTARG;;
        t) TRAIN=true;;
        f) export WOLFTEST_USE_FMA=false;;
        d) export WOLFTEST_USE_SQRTDIV=false;;
        e) export WOLFTEST_EPSILON=$OPTARG;;
        a) export WOLFTEST_EXHAUSTIVE=true;;
        h) printf "Usage: %s [-b] [-p platform] [-f] [-d] [-e epsilon] [root_folder]\n" $0
           printf " -b            build on the board, equivalent to -p board\n"
           printf " -p <platform> build on the desired platform [board | gvsoc], default is gvsoc\n"
           printf " -h            show this help message\n"
           printf " -f            do not use float fused multiply add instructions\n"
           printf " -d            do not use float divide and square root instructions\n"
           printf " -e  <epsilon> accuracy for all floating point tests, defaults to 1e-35\n"
           printf " -a            Do exhaustive testing of all the blocks in MRBCI"
           printf " root_folder   Start folder where to execute all the tests\n"
           exit 0;;
        ?) printf "Usage: %s [-b] [-p platform] root_folder\n" $0
           exit 2;;
    esac
done

# execute run.sh of the main folder
cd ..
printf "Building the main project...\n"
if [ "$TRAIN" = true ]; then
    bash run.sh -n -t
else
    bash run.sh -n > /dev/null
fi
cd test

printf "\n"

# search for a virtual environment
# if [ -e ../multiscale_bci_python/env/bin/activate ]; then
#     printf "activating multiscale_bci_python/env\n"
#     source ../multiscale_bci_python/env/bin/activate
# else
#     read -s "Please enter the path to a virtual environment (including /bin/activate)\n> " VIRTUAL_ENVIRONMENT_PATH
#     if [ -e $VIRTUAL_ENVIRONMENT_PATH ]; then
#         source $VIRTUAL_ENVIRONMENT_PATH
#     else
#         printf "Virtual environment could not be found!"
#         exit 1
#     fi
# fi

printf "Testing on Platform: %s" $PLATFORM
if [ "$WOLFTEST_USE_FMA" = true ]; then
    printf " +fma"
else
    printf " -fma"
fi
if [ "$WOLFTEST_USE_SQRTDIV" = true ]; then
    printf " +sqrtdiv"
else
    printf " -sqrtdiv"
fi
printf "\nepsilon = %s\n\n" $WOLFTEST_EPSILON

ROOT=${@:$OPTIND:1}

# setup environmental variables
export PYTHONPATH="$(pwd)/../python_utils:$(pwd)/../multiscale_bci_python:$PYTHONPATH"

# set the platform
export PULP_CURRENT_CONFIG_ARGS="platform=$PLATFORM"

# always store the trace file
# PULP_CURRENT_CONFIG_ARGS+=" gvsoc/trace=l2_priv:$(pwd)/../build/trace.txt"

# activate the environment for python3.8
source $CONDA_BASE_PREFIX/etc/profile.d/conda.sh
conda activate mrc

python3.8 run_test.py $ROOT

# deactivate pyhon3.8 env and go with python2.7 and python3.5/3.6 env
conda deactivate
