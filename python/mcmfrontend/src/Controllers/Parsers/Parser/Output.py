#!/usr/bin/env python3

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
from Models.EnumDeclarations import *
import Models.Layouts as Layouts
from Controllers.TensorFormat import TensorFormat


class Output(Layer):
    def __init__(self, *args):
        super().__init__(*args)
        self.isLayerDetectionOutput = False
        self.setLayoutIndependent()
        self.setImplicit()

        # Set the supported layouts
        tfCM = TensorFormat(Layouts.NHWC, (1, 2, 3))
        tfPL = TensorFormat(Layouts.NCHW, (1, 2, 3))
        tfIV = TensorFormat(Layouts.NHCW, (1, 2, 3))
        tfChannelMajor = TensorFormat(Layouts.ChannelMajor, (1, 2, 3))
        tfZMajor = TensorFormat(Layouts.ZMajor, (1, 2, 3))
        self.formatPool = [(tfCM, None), (tfPL, None), (tfIV, None),
                           (tfChannelMajor, None), (tfZMajor, None)]

    def enableDetectionOutput(self):
        self.isLayerDetectionOutput = True

    def isDetectionOutput(self):
        return self.isLayerDetectionOutput
