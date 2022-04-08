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
import Models.Layouts as Layouts
from Controllers.TensorFormat import TensorFormat
from Models.EnumDeclarations import PadStyle
import numpy as np

from enum import Enum


class Pooling(Layer):
    class Type(Enum):
        MAX = 'Maximum'
        AVE = 'Average'

    def __init__(self, *args):
        super().__init__(*args)

        # Set the supported layouts
        tfCM = TensorFormat(Layouts.NHWC, (2,))
        self.formatPool = [(tfCM, tfCM)]
        self.padStyle = PadStyle.caffe

    def getType(self):
        return self.type

    def loadType(self, type):
        self.type = type

    def loadGlobal(self, flag):
        self.globalPooling = flag

    def isGlobal(self):
        return self.globalPooling

    def getKernelSize(self):
        return (self.kernelHeight, self.kernelWidth)

    def getPadding(self):
        return (self.paddingHeight, self.paddingWidth)

    def getStride(self):
        return (self.strideHeight, self.strideWidth)

    def getStrideSize(self):
        return (self.strideHeight, self.strideWidth)

    def loadKernelSize(self, kernelHeight, kernelWidth):
        self.kernelHeight = kernelHeight
        self.kernelWidth = kernelWidth

    def loadStrideSize(self, strideHeight, strideWidth):
        self.strideHeight = strideHeight
        self.strideWidth = strideWidth

    def loadPadding(self, paddingHeight, paddingWidth):
        self.paddingHeight = paddingHeight
        self.paddingWidth = paddingWidth

    def setPadStyle(self, padStyle):
        if padStyle == b'VALID':
            self.padStyle = PadStyle.tfvalid
        elif padStyle == b'SAME':
            self.padStyle = PadStyle.tfsame

    def convert2DwConv(self):
        from Controllers.Parsers.Parser.Convolution2D import ConvolutionDepthWise2D
        from Controllers.Parsers.Parser.Layer import MangledName, OriginalName
        from Controllers.Tensor import PopulatedTensor
        from Controllers.Quantization import QuantizationParameters

        layer = ConvolutionDepthWise2D(
            self.name.stringifyOriginalName(), [], [])
        layer.setInputTensorsAllFields(self.getInputTensors())
        layer.setOutputTensorsAllFields(self.getOutputTensors())

        layer.loadStrideSize(self.strideHeight, self.strideWidth)
        layer.loadKernelSize(self.kernelHeight, self.kernelWidth)
        layer.loadPadding(self.paddingHeight, self.paddingWidth)
        layer.setPadStyle(self.padStyle)
        layer.setBiasEnabled(False)

        # FP16: Doing the divion operation by way of
        # post processing: bias, scale, shift
        # is proven to be better in terms of precision
        # than doing the division by way of FP16 weights data
        weights_data = np.ones(
            (1,
                self.getOutputTensors()[0].shape[1],
                self.kernelHeight,
                self.kernelWidth)).astype(
                self.getInputTensors()[0].dtype.nptype)
        weights_range = [(1, 1)] * len(weights_data.shape)

        layer.weights = PopulatedTensor(weights_data)
        layer.weights.name = MangledName(
            OriginalName(
                self.name.stringifyOriginalName() +
                "_weights"))

        # Scale division proved to be better than weights data
        scale = np.array(
            [1 / (self.kernelHeight * self.kernelWidth)]).astype(np.float32)
        zero_point = np.array([0]).astype(np.int32)
        layer.weights.setQuantizationParameters(QuantizationParameters(
            scale, zero_point, weights_data.dtype, weights_range))

        return layer
