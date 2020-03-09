# Description

This folder contains the necessary codes for thesting the functions. The main test script `run_test.py` searches recursively all files in the directiory `[project_root]/tests` for python scripts called `testcase.py`, and executes the function `testcase.test()`.

## Usage

For running the tests, it is important to change the current directory into the test folder:

```
cd test
./run_test [-b] [folder]
```

Then, you can execute the tests by running `./run_test.sh`. This script accepts some arguments. If no arguments provided, the script will run all tests on GVSOC. However, if you provide a relative path afterwards, it will only execute all tests which are found in this directory (and subdirectories, recursively). If you pass the parameter `-b`, the tests are executed on the board. See `./run_test.sh -h` for more information.

## `testcase.py`

This file contains all the information needed for a single testcase. The main function is the function `test()`, which is called by `run_test.py`. Before this function is executed, the current directory is changed to the location of `testcase.py`.The following should be done inside the `test()` function:

1. Generate the test vectors and compute the expected result
2. Generate a header file to be included in the test (`header_file.HeaderFile`)
3. Build the small test project, run it and pipe the output to a file (`make clean all run > result.out`)
4. Parse the file, and extract the results (`test_utils.parse_output`)
5. Print out the results of the test (`test_utils.TestLogger`)
6. Return a summary, containing the number of test cases executed and the number of successful tests.

Look at [example/testcase.py](example/testcase.py) to see an example of how the test can be used.

## Test Source

Tests consist of a small c project. For testing functions on the cluster, the following files are necessary:

- `test.c` containing the `main` function, run on the FC. The `main` function mounts the cluster and executes `cluster_entry`
- `cluster.c` contains the `cluster_entry` function, in which the data is loaded from the FC into local memory. Then, it starts the performance measurement, and executes the function to be tested. Afterwards, it checks if the results is correct, and prints the result to `stdout` (see [Parsing](README.md#parsing)).

## Parsing

To communicate the results from the PULP simulation runtime to python, we use `stdout`. The following format is used:

```## [ID]: [FIELD]: [VALUE]```

- `ID`: corresponds to a subcase within the c program. There can be n different subcases within a single test.
- `FIELD`: Field, currently, the following fields are allowed: `result`, `cycles`, `instructions`. 
- `VALUE`: The value of the corresponding field. For `result`, the value is either `OK` or `FAIL`. For `cycles` and `instructions`, the value is a number.


## Python Utils

To make testing easier, some Python Utils are provided

- `header_file.py`: This library allows you to quickly generate a c header file.
- `test_utils.py`: This library contains the parser (`test_utils.parse_output`) and the logger (`test_utils.TestLogger`), which prints the result of the test in an easy format.
