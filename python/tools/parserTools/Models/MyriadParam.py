# Copyright 2018 Intel Corporation.
# The source code, information and material ("Material") contained herein is
# owned by Intel Corporation or its suppliers or licensors, and title to such
# Material remains with Intel Corporation or its suppliers or licensors.
# The Material contains proprietary information of Intel or its suppliers and
# licensors. The Material is protected by worldwide copyright laws and treaty
# provisions.
# No part of the Material may be used, copied, reproduced, modified, published,
# uploaded, posted, transmitted, distributed or disclosed in any way without
# Intel's prior express written permission. No license under any patent,
# copyright or other intellectual property rights in the Material is granted to
# or conferred upon you, either expressly, by implication, inducement, estoppel
# or otherwise.
# Any license under such intellectual property rights must be express and
# approved by Intel in writing.

import numpy as np
from ctypes import *
from Controllers.MiscIO import *


class MyriadParam:
    def __init__(self, fs=0, ls=1, optimization_list=None):
        """
        Constructor for our myriad definitions
        :param fs: first shave to be used
        :param ls: last shave in range
        :return:
        """
        self.firstShave = c_ushort(fs)
        self.lastShave = c_ushort(ls)
        self.leonMemLocation = c_uint(0)
        self.leonMemSize = c_uint(0)
        self.dmaAgent = c_uint(0)
        self.optimization_list = optimization_list

    def generate(self, f):
        """
        Write to file.

        :param f:
        :return:
        """
        f.write(self.firstShave)
        f.write(self.lastShave)
        f.write(self.leonMemLocation)
        f.write(self.leonMemSize)
        f.write(self.dmaAgent)

    def binary_size(self):
        """
        get binary size of this element when written to file.
        :return:
        """
        file_size = byte_size(self.firstShave)
        assert file_size == 0x2, "Blob format modified, please change the " +\
                                 "FathomRun/tests/per_layer_tests/util/generate_test_data.py file"
        file_size += byte_size(self.lastShave)
        file_size += byte_size(self.leonMemLocation)
        file_size += byte_size(self.leonMemSize)
        file_size += byte_size(self.dmaAgent)
        return file_size

    def display_opts(self):
        print("\nAvailable Optimizations:")
        [print("* " + str(x)) for x in self.optimization_list]
