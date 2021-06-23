"""
This script searches recursively the current directory and executes all "gen_test.py" files.
"""

__author__ = "Tibor Schneider"
__email__ = "sctibor@student.ethz.ch"
__version__ = "0.1.0"
__date__ = "2020/01/19"
__copyright__ = """
    Copyright (C) 2020 ETH Zurich. All rights reserved.

    Author: Tibor Schneider, ETH Zurich

    SPDX-License-Identifier: Apache-2.0
    Licensed under the Apache License, Version 2.0 (the License); you may
    not use this file except in compliance with the License.
    You may obtain a copy of the License at

    www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an AS IS BASIS, WITHOUT
    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import os
import sys
import importlib.util

TEST_FILENAME = "testcase.py"

RED_COLOR = "\033[1;31m"
GREEN_COLOR = "\033[1;32m"
RESET_COLOR = "\033[0;0m"


def test_main(root_folder):
    """ main function """

    old_cwd = os.getcwd()

    """
    # go a directory up and build the project, without running it
    os.chdir("..")
    print("Building the project...")
    os.system("./run.sh -n > /dev/null")
    # go back to the test directory
    os.chdir(old_cwd)
    """

    # keep track of statistics
    num_total = 0
    num_success = 0

    # search recursively all files
    for root, _, files in sorted(os.walk(root_folder), key=lambda x: x[0]):
        for filename in files:
            if filename == TEST_FILENAME:
                # import the file
                spec = importlib.util.spec_from_file_location("testcase", os.path.join(root, filename))
                testcase = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(testcase)

                # change directory into the test directory
                os.chdir(root)

                # execute the test
                _num_total, _num_success = testcase.test()

                num_total += _num_total
                num_success += _num_success

                # leave directory again
                os.chdir(old_cwd)

    # print a summary
    all_ok = num_total == num_success
    print("\n********************")
    if all_ok:
        color = GREEN_COLOR
    else:
        color = RED_COLOR
    print("{}Summary: {}{}".format(color, ("OK" if all_ok else "FAIL"), RESET_COLOR))
    print("Number of tests: {}".format(num_total))
    print("Failed tests:    {}".format(num_total - num_success))


if __name__ == "__main__":
    ROOT_FOLDER = "."
    if len(sys.argv) > 1:
        ROOT_FOLDER = sys.argv[1]

    test_main(ROOT_FOLDER)
