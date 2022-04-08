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

from enum import Enum
import Models.Layouts as Layouts
from Controllers.TensorFormat import TensorFormat


class LRN(Layer):
    class Type(Enum):
        ACROSS = 'AcrossChannels'
        WITHIN = 'WithinChannel'
        # Myriad does a sub-section of InnerLRN, rather than the full operation
        WITHIN_PARTIAL = 'WithinChannelPartial'

    def __init__(self, *args):
        super().__init__(*args)
        self.addCompatibleLayout(Layouts.NCHW)    # Planar
        self.addCompatibleLayout(Layouts.NHCW)    # Row Interleaved

        # Set the supported layouts

        tfCM = TensorFormat(Layouts.NHWC, (2,))
        tfIV = TensorFormat(Layouts.NHCW, (1, 2, 3))
        self.formatPool = [(tfCM, tfCM), (tfIV, tfIV)]

    def getType(self):
        return self.type

    def getSquareKernelSize(self):
        return self.sideLength

    def getAlpha(self):
        return self.alpha

    def getBeta(self):
        return self.beta

    def getK(self):
        return self.k

    def loadType(self, type):
        self.type = type

    def loadSquareKernelSize(self, sideLength):
        self.sideLength = sideLength

    def loadAlpha(self, alpha):
        self.alpha = alpha

    def loadBeta(self, beta):
        self.beta = beta

    def loadK(self, k):
        self.k = k
