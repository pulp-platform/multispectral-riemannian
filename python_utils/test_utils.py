"""
These Utility functions enable easy unit tests for the EEGnet.
"""

__author__ = "Tibor Schneider"
__email__ = "sctibor@student.ethz.ch"
__version__ = "0.1.1"
__date__ = "2020/01/23"

import numpy as np

DOT_LENGTH = 40

RED_COLOR = "\033[1;31m"
GREEN_COLOR = "\033[1;32m"
RESET_COLOR = "\033[0;0m"

SUCCESS_STR = "..{}OK{}".format(GREEN_COLOR, RESET_COLOR)
FAIL_STR = "{}FAIL{}".format(RED_COLOR, RESET_COLOR)

def parse_output(filename):
    """
    This function parses the output of a test run.

    For each run of the test, the program should print the following:
        ## ID: result: [OK|FAIL]
        ## ID: cycles: N_CYCLES
        ## ID: instructions: N_INSTR
        ## ID: Key: Value

    Multiple runs are allowed.
    Make sure, that you pipe the execution into a file, whose name is then passed into this function.

    Example:
        os.system("make clean all run > result.out")
        parsed = parse_output("result.out")

    This function returns a dictionary in the form:
        { "1": {"result": "OK", "cycles": "215", "instructions": "201"}, ... }
    """
    parsed = {}
    with open(filename, "r") as _f:
        for line in _f.readlines():
            if not line.startswith("## "):
                continue
            line = line.lstrip("# ")
            parts = [p.strip() for p in line.split(":")]
            assert len(parts) == 3, line
            if parts[0] not in parsed:
                parsed[parts[0]] = {}
            if parts[1] == "result":
                parsed[parts[0]]["result"] = parts[2] == "OK"
            else:
                parsed[parts[0]][parts[1].lower()] = parts[2]

    # add IPC to the output
    for case in parsed.values():
        if "cycles" in case and "instructions" in case:
            case["ipc"] = "{:.3f}".format(int(case["instructions"]) / int(case["cycles"]))
    return parsed


class TestLogger:
    """
    Class to display the logging result
    """
    def __init__(self, name, show_title=True):
        self.name = name
        self.num_cases = 0
        self.num_successful = 0
        if show_title:
            print("\n**** Test Case: {}".format(self.name))

    def show_subcase_result(self, subcase_name, results):
        """
        Display the parsed results and count the number of (successful) test cases.

        Parameters:
        - subcase_name: str, name of the subcase to be displayed
        - results: parsed results file
        """
        assert results
        if len(results) == 1:
            result = list(results.values())[0]
            if result["result"] is None:
                success_str = "SKIP"
            else:
                success_str = SUCCESS_STR if result["result"] else FAIL_STR
            options = []
            for k in sorted(result):
                v = result[k]
                if k == "instructions":
                    k = "insn"
                if k == "result":
                    continue
                if isinstance(v, float) or isinstance(v, np.float32):
                    options.append("{}: {:.2E}".format(k, v))
                else:
                    options.append("{}: {}".format(k, v))
            options_str = ""
            if options:
                options_str = "[{}]".format(", ".join(options))
            print("{}{} {}" .format(subcase_name.ljust(DOT_LENGTH, "."), success_str, options_str))

            # keep track of statistics
            if result["result"] is not None:
                self.num_cases += 1
                if result["result"]:
                    self.num_successful += 1
        else:
            for case_id in sorted(results):
                result = results[case_id]
                if result["result"] is None:
                    success_str = "SKIP"
                else:
                    success_str = SUCCESS_STR if result["result"] else FAIL_STR
                options = []
                for k in sorted(result):
                    v = result[k]
                    if k == "result":
                        continue
                    if k == "instructions":
                        k = "insn"
                    if isinstance(v, float) or isinstance(v, np.float32):
                        options.append("{}: {:.2E}".format(k, v))
                    else:
                        options.append("{}: {}".format(k, v))
                options_str = ""
                if options:
                    options_str = "[{}]".format(", ".join(options))
                subcase_str = "{} {}".format(subcase_name, case_id)
                print("{}{} {}" .format(subcase_str.ljust(DOT_LENGTH, "."), success_str, options_str))

                # keep track of statistics
                if result["result"] is not None:
                    self.num_cases += 1
                    if result["result"]:
                        self.num_successful += 1

    def summary(self):
        """
        Returns tuple: (number of test cases, number of successful test cases)
        """
        return self.num_cases, self.num_successful
