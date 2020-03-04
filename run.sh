#! /bin/bash

PLATFORM="gvsoc"
RUN=true
GTKWAVE=false
TRAIN=false

while getopts "bp:nwtfde:h" name; do
    case "$name" in
        b) PLATFORM="board";;
        p) PLATFORM=$OPTARG;;
        n) RUN=false;;
        w) GTKWAVE=true;;
        t) TRAIN=true;;
        h) printf "Usage: %s [-b] [-p platform] [-h] [-n] [-w]\n" $0
           printf " -b            build on the board, equivalent to -p board\n"
           printf " -p <platform> build on the desired platform [board | gvsoc], default is gvsoc\n"
           printf " -n            do not run the program, just build it\n"
           printf " -w            generate GTK wave files\n"
           printf " -h            show this help message\n"
           exit 0;;
        ?) printf "Usage: %s [-b] [-p platform] root_folder\n" $0
           exit 2;;
    esac
done


if [ "$TRAIN" = true ]; then
    printf "Training the network...\n\n"

    # enter python bci directory
    cd multiscale_bci_python

    # train the network (for one subject only) and export the data to the ../data directory
    python3.8 main_riemannian.py -e -f ../data

    # go back to the root hdirectory
    cd ..
fi

printf "Running EEGnet on Platform: %s\n\n" $PLATFORM

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

# generate net header file
python3.8 generate_mrbci_header.py
python3.8 generate_input_header.py

# leave data directory
cd ..

# deactivate the virtual environment and reset the pythonpath
# deactivate

# build everything
make clean all

# run if requested
if [ "$GTKWAVE" = true ] ; then
    make run runner_args="--vcd --event=.*"
else
    if [ "$RUN" = true ] ; then
        make run
    fi
fi
