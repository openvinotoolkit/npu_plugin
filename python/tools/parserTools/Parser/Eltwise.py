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
from parserTools.Parser.Concat import Concat
from parserTools.Parser.Convolution2D import Convolution2D
from parserTools.Tensor import UnpopulatedTensor
import parserTools.Models.Layouts as Layouts
from parserTools.TensorFormat import TensorFormat
from parserTools.Parser.Layer import MangledName, OriginalName
import numpy as np

from enum import Enum

class Eltwise(Layer):
    class Type(Enum):
        WMAX = 'WeightedMaximum'
        WPROD = 'WeightedProduct'
        WSUM = 'WeightedSum'

    def __init__(self, *args):
        super().__init__(*args)

        # Set the supported layouts
        tfCM = TensorFormat(Layouts.NHWC, (2,), 0, (1, 1, 1, 1))
        tfIV = TensorFormat(Layouts.NHCW, (3,), 0, (1, 1, 1, 1))

        # TODO: is this the right solution to prioritize the chosen layout ?

        self.formatPool = [(tfCM, tfCM), (tfIV, tfIV)]

        self.hasInplaceReLU = False
        self.reLUnegSlope = 0.0
        self.reLUposSlope = 1.0

    def loadType(self, type):
        self.type = type

        # Default coefficient treats the operation as unweighted
        self.coefficients = [1.0] * self.getInputTensorsCount()

    def getType(self):
        return self.type

    def loadCoefficients(self, coeff):
        assert(len(coeff) == len(self.coefficients))
        self.coefficients = list(coeff)

    def getStrideSize(self):
        return (1, 1)

    def getKernelSize(self):
        return (1, 1)

    def getPadding(self):
        return [(0, 0), (0, 0)]

    def convert2Conv(self):

        concat_tensor_name = MangledName(OriginalName(self.getOutputTensors()[0].name.stringifyOriginalName()))
        concat_tensor_channel = np.sum([self.getInputTensors()[i].shape[1] for i in range(len(self.getInputTensors()))])
        concat_tensor_shape = (self.getOutputTensors()[0].shape[0], int(concat_tensor_channel), self.getOutputTensors()[0].shape[2], self.getOutputTensors()[0].shape[3])

        concat_tensor = UnpopulatedTensor(concat_tensor_shape)
        concat_tensor.setName(concat_tensor_name)

        concat = Concat(self.name.stringifyOriginalName() + "_concat", self.inputTensorNames, [concat_tensor_name])
        concat.setInputTensors(self.getInputTensors())
        concat.loadInputTensorSizes(self.getInputTensorSizes())
        concat.setOutputTensors([concat_tensor])
        concat.loadOutputTensorSizes([concat_tensor_shape])

        conv = Convolution2D(self.name.stringifyOriginalName() + "_conv", [concat_tensor_name], self.outputTensorNames)
        conv.setInputTensors([concat_tensor])
        conv.loadInputTensorSizes([concat_tensor_shape])
        conv.setOutputTensors(self.getOutputTensors())
        conv.loadOutputTensorSizes(self.getOutputTensorSizes())

        conv.loadKernelSize(1, 1)
        conv.loadPadding(0,0)
        conv.loadDilation(1)
        conv.loadStrideSize(1, 1)
        conv.setBiasEnabled(False)
        conv.loadGroupSize(1)

        '''
        The convolution's kernel is the concatenation of two identity matrix (CxC) in the C dimension
        This allows to sum toghether in an element-wise fashion each input channel separately
        '''
        # k, c, kh, kw
        kernels = []
        for i, tensor in enumerate(self.getInputTensors()):
            kernels.append(self.coefficients[i]*np.eye(tensor.shape[1]))

        kernel = np.expand_dims(np.expand_dims(np.transpose(np.vstack(kernels)), -1), -1)
        conv.setWeights(kernel)

        return conv, concat
