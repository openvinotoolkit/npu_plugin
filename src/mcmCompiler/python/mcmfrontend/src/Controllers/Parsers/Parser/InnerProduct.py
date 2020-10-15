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
from Controllers.Tensor import PopulatedTensor
import Models.Layouts as Layouts
from Controllers.TensorFormat import TensorFormat
from Controllers.Parsers.Parser.Layer import MangledName, OriginalName
from Controllers.Quantization import QuantizationParameters


class InnerProduct(Layer):

    def __init__(self, *args):
        super().__init__(*args)

        # Set the supported layouts
        tfCM = TensorFormat(Layouts.NHWC, (1,))
        self.formatPool = [(tfCM, tfCM)]

    def loadKernelSize(self, kernelHeight, kernelWidth):
        self.kernelHeight = kernelHeight
        self.kernelWidth = kernelWidth

    def loadStrideSize(self, strideHeight, strideWidth):
        self.strideHeight = strideHeight
        self.strideWidth = strideWidth

    def loadPadding(self, paddingHeight, paddingWidth):
        self.paddingHeight = paddingHeight
        self.paddingWidth = paddingWidth

    def loadDilation(self, dilationFactor):
        self.dilationFactor = dilationFactor

    def setBiasEnabled(self, flag):
        self.hasBias = flag

    def biasEnabled(self):
        return self.hasBias

    def getPadding(self):
        return (self.paddingHeight, self.paddingWidth)

    def getStrideSize(self):
        return (self.strideHeight, self.strideWidth)

    def getDilation(self):
        return self.dilationFactor

    def getKernelSize(self):
        return self.kernelHeight, self.kernelWidth

    def getBias(self):
        return self.bias

    def getWeights(self):
        return self.weights

    def loadOutputChannels(self, outputChannels):
        self.outputChannels = outputChannels

    def setBias(self, bias):
        self.bias = PopulatedTensor(bias)
        self.bias.name = MangledName(
            OriginalName(
                self.name.stringifyOriginalName() +
                "_bias"))

    def setWeights(self, weights):
        self.weights = PopulatedTensor(weights)
        self.weights.setLayout((0, 3, 2, 1))
        self.weights.name = MangledName(
            OriginalName(
                self.name.stringifyOriginalName() +
                "_weights"))

    def loadTrainedParameters(self, **kwargs):
        self.setWeights(kwargs['weights'])
        bias = kwargs['bias']
        if bias is not None:
            self.setBias(bias)
        else:
            print("No Bias")

    def flatten(self):
        """
            By flattening an Inner Product, we can operate on Vectors rather
            than Matricies.
        """

        w = self.getWeights().data

        w = w.flatten()

        self.loadTrainedParameters(weights=w)

    def isInput3D(self):
        s = self.inputTensors[0].getTopEncloserRecursive().getShape()
        return len(s) - 1 != s.count(1)

    def canBeConvolution(self):
        # Check if input volume is 3D
        return self.isInput3D()

    def setQuantizationParameters(self, weight_q, bias_q=None):

        tensor_range = [(weight_q.Min(idx), weight_q.Max(idx))
                        for idx in range(weight_q.MinLength())]
        self.weights.setQuantizationParameters(
            QuantizationParameters(
                weight_q.ScaleAsNumpy(),
                weight_q.ZeroPointAsNumpy(),
                self.weights.dtype,
                tensor_range))

        if self.biasEnabled():
            tensor_range = [(bias_q.Min(idx), bias_q.Max(idx))
                            for idx in range(bias_q.MinLength())]
            self.bias.setQuantizationParameters(
                QuantizationParameters(
                    bias_q.ScaleAsNumpy(),
                    bias_q.ZeroPointAsNumpy(),
                    self.bias.dtype,
                    tensor_range))

    def convert2Conv(self):
        from Controllers.Parsers.Parser.Convolution2D import Convolution2D

        layer = Convolution2D(
            self.getStringifiedName(),
            self.inputTensorNames,
            self.outputTensorNames)

        # Copy the attributes
        for attr_name in self.__dict__:
            setattr(layer, attr_name, getattr(self, attr_name))

        out_ch = self.outputTensors[0].getTopEncloserRecursive().getShape()[1]
        _, in_ch, height, width = self.inputTensors[0].getTopEncloserRecursive(
        ).getShape()
        layer.weights.reshape((out_ch, in_ch, height, width))

        # Set kernel parameter
        layer.loadKernelSize(height, width)
        layer.loadStrideSize(1, 1)
        layer.loadPadding((0, 0), (0, 0))
        layer.loadGroupSize(1)
        layer.loadDilation(1)

        return layer
