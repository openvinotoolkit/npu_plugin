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


from .Layer import Layer
import Models.Layouts as Layouts
from Controllers.TensorFormat import TensorFormat
from enum import Enum


class Pad(Layer):
    class PadType(Enum):
        CONST = 0
        AVGPOOL = 1

        def __int__(self):
            return self.value

    def __init__(self, *args):
        super().__init__(*args)
        tfCM = TensorFormat(Layouts.NHWC, (2,))
        tfIV = TensorFormat(Layouts.NHCW, (1, 3))
        self.formatPool = [(tfCM, tfCM), (tfIV, tfIV)]

    def set_padding_size(self, pad_top, pad_bottom, pad_left, pad_right):
        self.pad_top = pad_top
        self.pad_bottom = pad_bottom
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.pad_type = Pad.PadType.CONST
        self.pad_const = 0.0

    def get_padding_size(self):
        return [[self.pad_top, self.pad_bottom],
                [self.pad_left, self.pad_right]]

    def set_padding_type(self, pad_type):
        self.pad_type = pad_type

    def get_padding_type(self):
        return self.pad_type

    def compute_output_shape(self):
        assert(len(self.inputTensorSizes) == 1)

        in_batch, in_channels, in_height, in_width = self.inputTensorSizes[0]

        out_batch = in_batch
        out_channels = in_channels
        out_width = in_width + self.pad_left + self.pad_right
        out_height = in_height + self.pad_top + self.pad_bottom

        self.outputTensorSizes = [
            [out_batch, out_channels, out_height, out_width]]
