"""
Class to geerate C Header Files
"""

__author__ = "Tibor Schneider"
__email__ = "sctibor@student.ethz.ch"
__version__ = "0.1.1"
__date__ = "2020/01/28"

import os
import re
from textwrap import wrap
import numpy as np
from collections import OrderedDict

MAX_WIDTH = 80
TAB = "    "


class HeaderFile():
    """
    Enables comfortable generation of header files
    if with_c is set, then generate a c file of the same name, but with a .c ending.
    """
    def __init__(self, filename, define_guard=None, with_c=False):
        assert filename.endswith(".h")
        self.filename = filename
        self.with_c = with_c
        if with_c:
            self.c_filename = self.filename.rstrip("h") + "c"
        self.define_guard = define_guard
        if self.define_guard is None:
            self.define_guard = "__" + re.sub("[./]", "_", filename.upper()) + "__"
        self.elements = []

    def add(self, element):
        self.elements.append(element)

    def header_str(self):
        ret = ""
        ret += "#ifndef {}\n".format(self.define_guard)
        ret += "#define {}\n\n".format(self.define_guard)
        ret += "#include \"rt/rt_api.h\"\n\n"

        for element in self.elements:
            ret += element.header_str(self.with_c)

        ret += "#endif//{}".format(self.define_guard)
        return ret

    def source_str(self):
        assert self.with_c
        ret = ""
        ret += "#include \"{}\"\n\n".format(os.path.split(self.filename)[-1])
        for element in self.elements:
            ret += element.source_str()
        return ret

    def write(self):
        with open(self.filename, "w") as _f:
            _f.write(self.header_str())
        if self.with_c:
            with open(self.c_filename, "w") as _f:
                _f.write(self.source_str())


class HeaderEntry():
    def __init__(self):
        pass

    def header_str(self, with_c=False):
        return ""

    def source_str(self):
        return ""


class HeaderStruct(HeaderEntry):
    """ Adds a struct to the header (and source) file """
    def __init__(self, name, struct_type, data, blank_line=True):
        assert isinstance(data, (dict, OrderedDict))
        self.name = name
        self.struct_type = struct_type
        self.data = data
        self.blank_line = blank_line

    def header_str(self, with_c=False):
        if with_c:
            ret = "extern const {} {};\n".format(self.struct_type, self.name)
            if self.blank_line:
                ret += "\n"
        else:
            ret = self.source_str()
        return ret

    def source_str(self):
        # first, compute the length of the longest value
        max_len = max([len(str(v)) for v in self.data.values()])
        max_len += 1

        ret = "const {} {} = {};\n".format(self.struct_type, self.name, self.initializer_str())
        if self.blank_line:
            ret += "\n"
        return ret

    def initializer_str(self, double_tab=False):
        max_len = max([len(str(v)) for v in self.data.values()])
        max_len += 1

        item_tab = TAB
        if double_tab:
            item_tab = TAB + TAB

        ret = "{\n"
        for key, value in self.data.items():
            val_colon = "{},".format(value).ljust(max_len, ' ')
            ret += "{}{} //{}\n".format(item_tab, val_colon, key)
        if double_tab:
            ret += "{}}}".format(TAB)
        else:
            ret += "}"
        return ret


class HeaderConstant(HeaderEntry):
    def __init__(self, name, value, blank_line=True):
        self.name = name
        self.value = value
        self.blank_line = blank_line

    def header_str(self, with_c=False):
        ret = "#define {} {}\n".format(self.name, self.value)
        if self.blank_line:
            ret += "\n"
        return ret

    def source_str(self):
        return ""


class HeaderInclude(HeaderEntry):
    def __init__(self, name, blank_line=True):
        self.name = name
        self.blank_line = blank_line

    def header_str(self, with_c=False):
        ret = "#include \"{}\"\n".format(self.name)
        if self.blank_line:
            ret += "\n"
        return ret

    def source_str(self):
        return ""


class HeaderScalar(HeaderEntry):
    def __init__(self, name, dtype, value, const=True, blank_line=True):
        self.name = name
        self.dtype = dtype
        self.value = value
        self.const = const
        self.const_str = "const " if self.const else ""
        self.blank_line = blank_line

    def header_str(self, with_c=False):
        if with_c:
            ret = "extern {}{} {};\n".format(self.const_str, self.dtype, self.name, self.value)
        else:
            ret = "{}{} {} = {};\n".format(self.const_str, self.dtype, self.name, self.value)

        if self.blank_line:
            ret += "\n"
        return ret

    def source_str(self):
        ret = "{}{} {} = {};\n".format(self.const_str, self.dtype, self.name, self.value)
        if self.blank_line:
            ret += "\n"
        return ret


class HeaderArray(HeaderEntry):
    def __init__(self, name, dtype, data, locality="RT_L2_DATA", blank_line=True, const=True,
                 formatter=str, skip_format=False):
        assert locality in ["RT_LOCAL_DATA", "RT_L2_DATA", "RT_CL_DATA", "RT_FC_SHARED_DATA",
                            "RT_FC_GLOBAL_DATA", ""]
        self.name = name
        self.dtype = dtype
        self.data = data
        self.locality = locality
        self.const = const
        self.blank_line = blank_line
        self.formatter = formatter
        self.skip_format = skip_format

    def header_str(self, with_c=False):
        const_str = "const " if self.const else ""
        if with_c:
            ret = "extern {} {}{} {}[{}];\n".format(self.locality, const_str, self.dtype, self.name, len(self.data))
        else:
            # first, try it as a one-liner (only if the length is smaller than 16)
            if len(self.data) <= 16:
                ret = "{} {}{} {}[] = {{ {} }};".format(self.locality, const_str, self.dtype, self.name,
                                                        ", ".join([self.formatter(item) for item in self.data]))
                if len(ret) <= MAX_WIDTH:
                    ret += "\n"
                    if self.blank_line:
                        ret += "\n"
                    return ret

            # It did not work on one line. Make it multiple lines
            ret = ""
            ret += "{} {}{} {}[] = {{\n".format(self.locality, const_str, self.dtype, self.name)

            if self.skip_format:
                ret += TAB
                ret += ",\n{}".format(TAB).join(self.data)
            else:
                long_str = ", ".join([self.formatter(item) for item in self.data])
                parts = wrap(long_str, MAX_WIDTH-len(TAB))
                ret += "{}{}".format(TAB, "\n{}".format(TAB).join(parts))

            ret += "\n};\n"

        if self.blank_line:
            ret += "\n"
        return ret

    def source_str(self):
        const_str = "const " if self.const else ""
        # first, try it as a one-liner
        if len(self.data) <= 16:
            ret = "{} {}{} {}[] = {{ {} }};".format(self.locality, const_str, self.dtype, self.name,
                                                    ", ".join([self.formatter(item) for item in self.data]))
            if len(ret) <= MAX_WIDTH:
                ret += "\n"
                if self.blank_line:
                    ret += "\n"
                return ret

        # It did not work on one line. Make it multiple lines
        ret = ""
        ret += "{} {}{} {}[] = {{\n".format(self.locality, const_str, self.dtype, self.name)

        if self.skip_format:
            ret += TAB
            ret += ",\n{}".format(TAB).join(self.data)
        else:
            long_str = ", ".join([self.formatter(item) for item in self.data])
            parts = wrap(long_str, MAX_WIDTH-len(TAB))
            ret += "{}{}".format(TAB, "\n{}".format(TAB).join(parts))

        ret += "\n};\n"
        if self.blank_line:
            ret += "\n"
        return ret


class HeaderComment(HeaderEntry):
    def __init__(self, text, mode="//", blank_line=True):
        assert mode in ["//", "/*"]
        self.text = text
        self.mode = mode
        self.blank_line = blank_line

    def header_str(self, with_c=False):
        if self.mode == "//":
            start = "// "
            mid = "\n// "
            end = ""
        else:
            start = "/*\n * "
            mid = "\n * "
            end = "\n */"
        ret = start
        ret += mid.join([mid.join(wrap(par, MAX_WIDTH-3)) for par in self.text.split("\n")])
        ret += end
        ret += "\n"
        if self.blank_line:
            ret += "\n"
        return ret

    def source_str(self):
        return self.header_str(True)


def align_array(x, n=4, fill=0):
    """
    Aligns the array to n elements (i.e. Bytes) by inserting fill bytes

    Parameters:
    - x: np.array(shape: [..., D])
    - n: number of elements to be aligned with
    - fill: value of the added elements

    Return: np.array(shape: [..., D']), where D' = ceil(D / n) * n
    """

    new_shape = (*x.shape[:-1], align_array_size(x.shape[-1]))
    y = np.zeros(new_shape, dtype=x.dtype)

    original_slice = tuple(slice(0, d) for d in x.shape)
    fill_slice = (*tuple(slice(0, d) for d in x.shape[:-1]), slice(x.shape[-1], y.shape[-1]))

    y[original_slice] = x
    y[fill_slice] = fill

    return y


def align_array_size(D, n=4):
    """
    Computes the required dimension D' such that D' is aligned and D <= D'

    Parameters:
    - D: Dimension to be aligned
    - n: number of elements to be aligned with

    Return: D'
    """
    return int(np.ceil(D / n) * n)
