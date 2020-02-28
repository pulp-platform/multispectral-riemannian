"""
Class to generate a Makefile
"""

__author__ = "Tibor Schneider"
__email__ = "sctibor@student.ethz.ch"
__version__ = "0.1.0"
__date__ = "2020/01/29"

import os

FILES_IN_ROOT = {'data', 'src', 'README.md', 'test', 'Makefile', 'python_utils', '.gitignore',
                 'run.sh', 'multiscale_bci_python', '.gitmodules'}


class Makefile:
    """ Makefile generation """
    def __init__(self, project_root=None, use_dsp=True, use_fma=None, use_sqrtdiv=None):
        self.fc_sources = []
        self.cl_sources = []
        self.defines = []
        self.use_dsp = use_dsp
        self.use_fma = use_fma
        self.use_sqrtdiv = use_sqrtdiv
        self.project_root = project_root
        if self.project_root is None:
            # search the project root
            self.project_root = os.getcwd()
            current_files = set(os.listdir(self.project_root))
            while not FILES_IN_ROOT <= current_files:
                # go in parent directory
                self.project_root = os.path.realpath(os.path.join(self.project_root, ".."))
                current_files = set(os.listdir(self.project_root))
                assert self.project_root != "/"
        else:
            current_files = set(os.listdir(self.project_root))
            assert FILES_IN_ROOT <= current_files

        # if use_fma or use_sqrt_div are not set, get the environmental variables
        if self.use_fma is None:
            self.use_fma = os.environ["WOLFTEST_USE_FMA"] == "true"
        if self.use_sqrtdiv is None:
            self.use_sqrtdiv = os.environ["WOLFTEST_USE_SQRTDIV"] == "true"

    def add_fc_test_source(self, name):
        """ add test source file, located in current directory """
        self.fc_sources.append(name)

    def add_cl_test_source(self, name):
        """ add test source file, located in current directory """
        self.cl_sources.append(name)

    def add_fc_prog_source(self, name):
        """ add source file from the actual program, starting at root/src/fc/ """
        source_file = os.path.join(self.project_root, "src/fc", name)
        assert os.path.exists(source_file), "Could not find FC source file: {}".format(source_file)
        assert source_file.endswith(".c")
        self.cl_sources.append(source_file)

    def add_cl_prog_source(self, name):
        """ add source file from the actual program, starting at root/src/cl/ """
        source_file = os.path.join(self.project_root, "src/cl", name)
        assert os.path.exists(source_file), "Could not find CL source file: {}".format(source_file)
        assert source_file.endswith(".c")
        self.cl_sources.append(source_file)

    def add_define(self, name, value=None):
        """ Those defines will be passed to gcc with -Dname=value flag """
        assert name.isupper()
        if value is None:
            self.defines.append(name)
        else:
            self.defines.append("{}={}".format(name, value))

    def __str__(self):
        ret = ""
        ret += "PULP_APP = test\n\n"

        # add cl sources
        if self.cl_sources:
            ret += "PULP_APP_CL_SRCS = \\\n"
            ret += "\n".join(["    {} \\".format(name) for name in self.cl_sources])
            ret += "\n\n"

        # add fc sources
        if self.fc_sources:
            ret += "PULP_APP_FC_SRCS = \\\n"
            ret += "\n".join(["    {} \\".format(name) for name in self.fc_sources])
            ret += "\n\n"

        # link dsp library
        if self.use_dsp:
            ret += "PULP_LDFLAGS += -lplpdsp\n"

        ret += "PULP_CFLAGS = -O3 -g \n\n"
        # link math library
        ret += "PULP_LDFLAGS += -lm\n\n"

        # Enable FMA
        if self.use_fma:
            ret += "# Enable Fused FMA\n"
            ret += "PULP_CFLAGS += -DUSE_FUSED_FPU\n\n"

        # Disable div sqrt
        if not self.use_sqrtdiv:
            ret += "# Disable SQRT/DIV unit\n"
            ret += "PULP_CFLAGS += -DUSE_SOFT_SQRTDIV\n"
            ret += "PULP_CFLAGS += -mno-fdiv\n"

        # add compiler flags
        ret += "\n".join(["PULP_CFLAGS += -D{}".format(define) for define in self.defines])
        ret += "\n\n"

        # include the pulp sdk
        ret += "include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk\n"
        return ret

    def write(self):
        """ write Makefile, while deleting the existing one """
        filename = "Makefile"
        if os.path.exists(filename):
            os.remove(filename)

        with open(filename, "w") as _f:
            _f.write(str(self))
