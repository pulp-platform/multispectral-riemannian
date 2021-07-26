#! /bin/bash
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
RUN=true
GTKWAVE=false
TRAIN=false
CHIP="wolfe"

while getopts "bvp:nwtfde:h" name; do
    case "$name" in
        b) PLATFORM="board";;
        v) CHIP="vega";;
        p) PLATFORM=$OPTARG;;
        n) RUN=false;;
        w) GTKWAVE=true;;
        t) TRAIN=true;;
        h) printf "Usage: %s [-b] [-p platform] [-h] [-n] [-w]\n" $0
           printf " -b            build on the board, equivalent to -p board\n"
           printf " -p <platform> build on the desired platform [board | gvsoc], default is gvsoc\n"
           printf " -v            use vega instead of mr. wolf\n"
           printf " -n            do not run the program, just build it\n"
           printf " -w            generate GTK wave files\n"
           printf " -h            show this help message\n"
           exit 0;;
        ?) printf "Usage: %s [-b] [-p platform] root_folder\n" $0
           exit 2;;
    esac
done

# to be able to use conda activate in bash scripts
source $CONDA_BASE_PREFIX/etc/profile.d/conda.sh

if [ "$TRAIN" = true ]; then
    printf "Training the network...\n\n"

    # enter python bci directory
    cd multiscale_bci_python

    # activate the environment for python3.8
    conda activate mrc

    # train the network (for one subject only) and export the data to the ../data directory
    python3.8 main_riemannian.py -e -f ../data

    # deactivate pyhon3.8 env and go with python2.7 and python3.5/3.6 env
    conda deactivate

    # go back to the root hdirectory
    cd ..
fi

printf "Running MRC on Platform: %s\n\n" $PLATFORM

# set the platform
PULP_CURRENT_CONFIG_ARGS="platform=$PLATFORM"

# build python utils
cd python_utils
make
cd ..

# add python_utils to the python path
export PYTHONPATH=$(pwd)/python_utils:$(pwd)/multiscale_bci_python:$PYTHONPATH

# enter data directory
cd data

conda activate mrc

# generate net header file
python3.8 generate_mrbci_header.py
python3.8 generate_input_header.py

conda deactivate

# leave data directory
cd ..

# deactivate the virtual environment and reset the pythonpath
# deactivate

# build everything
if [ "$CHIP" = "vega" ] ; then
    make clean all PMSIS_OS=pulp-os platform=gvsoc
else
    make clean all
fi

# run if requested
if [ "$GTKWAVE" = true ] ; then
    make run runner_args="--vcd --event=.*"
else
    if [ "$RUN" = true ] ; then
        if [ "$CHIP" = "vega" ] ; then
            make run platform=gvsoc
        else
            make run
        fi
    fi
fi
