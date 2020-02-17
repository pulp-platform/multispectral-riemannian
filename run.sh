#! /bin/bash

PLATFORM="gvsoc"
RUN=true
GTKWAVE=false
TRAIN=false

while getopts "bp:nwth" name; do
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

    # search for a virtual environment
    if [ -e env/bin/activate ]; then
        printf "activating multiscale_bci_python/env\n"
        source env/bin/activate
    elif [ -e venv/bin/activate ]; then
        printf "activating multiscale_bci_python/venv\n"
        source env/bin/activate
    elif [ -e virtualenv/bin/activate ]; then
        printf "activating multiscale_bci_python/virtualenv\n"
        source env/bin/activate
    else
        read -s "Please enter the path to a virtual environment (including /bin/activate)\n> " VIRTUAL_ENVIRONMENT_PATH
        if [ -e $VIRTUAL_ENVIRONMENT_PATH ]; then
            source $VIRTUAL_ENVIRONMENT_PATH
        else
            printf "Virtual environment could not be found!"
            exit 1
        fi
    fi

    # train the network (for one subject only) and export the data to the ../data directory
    python main_riemannian.py -e -f ../data

    # go back to the root hdirectory
    cd ..
fi

# TODO remove when some c code exists
exit 0

printf "Running EEGnet on Platform: %s\n\n" $PLATFORM

# add python_utils to the python path
export PYTHONPATH=$PYTHONPATH:$(pwd)/python_utils:$(pwd)/multiscale_bci_python

# set the platform
PULP_CURRENT_CONFIG_ARGS="platform=$PLATFORM"

# enter data directory
cd data

# generate net header file
python3 gen_net_header.py
python3 gen_input_header.py

# leave data directory
cd ..

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
