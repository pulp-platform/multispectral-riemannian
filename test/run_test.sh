#!/bin/bash

PLATFORM="gvsoc"
TRAIN=false
export WOLFTEST_USE_FMA=true
export WOLFTEST_USE_SQRTDIV=true
export WOLFTEST_EPSILON=1e-35

while getopts "bp:tfde:h" name; do
    case "$name" in
        b) PLATFORM="board";;
        p) PLATFORM=$OPTARG;;
        t) TRAIN=true;;
        f) export WOLFTEST_USE_FMA=false;;
        d) export WOLFTEST_USE_SQRTDIV=false;;
        e) export WOLFTEST_EPSILON=$OPTARG;;
        h) printf "Usage: %s [-b] [-p platform] [-f] [-d] [-e epsilon] [root_folder]\n" $0
           printf " -b            build on the board, equivalent to -p board\n"
           printf " -p <platform> build on the desired platform [board | gvsoc], default is gvsoc\n"
           printf " -h            show this help message\n"
           printf " -f            do not use float fused multiply add instructions\n"
           printf " -d            do not use float divide and square root instructions\n"
           printf " -a  <epsilon> accuracy for all floating point tests, defaults to 1e-35\n"
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

python3.8 run_test.py $ROOT
