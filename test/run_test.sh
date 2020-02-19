#!/bin/bash

PLATFORM="gvsoc"
TRAIN_FLAG="> /dev/null"

while getopts "bp:th" name; do
    case "$name" in
        b) PLATFORM="board";;
        p) PLATFORM=$OPTARG;;
        t) TRAIN_FLAG="-t";;
        h) printf "Usage: %s [-b] [-p platform] [root_folder]\n" $0
           printf " -b            build on the board, equivalent to -p board\n"
           printf " -p <platform> build on the desired platform [board | gvsoc], default is gvsoc\n"
           printf " -h            show this help message\n"
           printf " root_folder   Start folder where to execute all the tests\n"
           exit 0;;
        ?) printf "Usage: %s [-b] [-p platform] root_folder\n" $0
           exit 2;;
    esac
done

# execute run.sh of the main folder
cd ..
bash run.sh -n $TRAIN_FLAG
cd test

printf "\n"

# search for a virtual environment
if [ -e ../multiscale_bci_python/env/bin/activate ]; then
    printf "activating multiscale_bci_python/env\n"
    source ../multiscale_bci_python/env/bin/activate
else
    read -s "Please enter the path to a virtual environment (including /bin/activate)\n> " VIRTUAL_ENVIRONMENT_PATH
    if [ -e $VIRTUAL_ENVIRONMENT_PATH ]; then
        source $VIRTUAL_ENVIRONMENT_PATH
    else
        printf "Virtual environment could not be found!"
        exit 1
    fi
fi

printf "Testing on Platform: %s\n\n" $PLATFORM

ROOT=${@:$OPTIND:1}

# setup environmental variables
export PYTHONPATH="$PYTHONPATH:$(pwd)/../python_utils:$(pwd)/../multiscale_bci_python"

# set the platform
export PULP_CURRENT_CONFIG_ARGS="platform=$PLATFORM"

# always store the trace file
# PULP_CURRENT_CONFIG_ARGS+=" gvsoc/trace=l2_priv:$(pwd)/../build/trace.txt"

python3 run_test.py $ROOT
