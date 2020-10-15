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
from Controllers.Tensor import UnpopulatedTensor, PopulatedTensor
from Controllers.TensorFormat import TensorFormat
from Models.EnumDeclarations import PadStyle
from Controllers.Parsers.Parser.Upsampling import Upsampling
from Controllers.Parsers.Parser.crop import Crop
from Controllers.Parsers.Parser.Layer import MangledName, OriginalName
from Controllers.Quantization import QuantizationParameters
import Models.Layouts as Layouts
import numpy as np


class Convolution2DUniversal(Layer):
    def __init__(self, *args):
        super().__init__(*args)
        self.padStyle = PadStyle.caffe

    def loadKernelSize(self, kernelHeight, kernelWidth):
        self.kernelHeight = kernelHeight
        self.kernelWidth = kernelWidth

    def getKernelSize(self):
        return self.kernelHeight, self.kernelWidth

    def loadStrideSize(self, strideHeight, strideWidth):
        self.strideHeight = strideHeight
        self.strideWidth = strideWidth

    def getStrideSize(self):
        return (self.strideHeight, self.strideWidth)

    def loadPadding(self, paddingHeight, paddingWidth):
        self.paddingHeight = paddingHeight
        self.paddingWidth = paddingWidth

    def setPadStyle(self, padStyle):
        if padStyle == 'VALID':
            self.padStyle = PadStyle.tfvalid
        elif padStyle == 'SAME':
            self.padStyle = PadStyle.tfsame

    def getPadding(self):
        return (self.paddingHeight, self.paddingWidth)

    def loadDilation(self, dilationFactor):
        self.dilationFactor = dilationFactor

    def getDilation(self):
        return self.dilationFactor

    def setBiasEnabled(self, flag):
        self.hasBias = flag

    def biasEnabled(self):
        return self.hasBias

    def getBias(self):
        return self.bias

    def setBias(self, data):
        self.bias = PopulatedTensor(data)
        self.bias.name = MangledName(
            OriginalName(
                self.name.stringifyOriginalName() +
                "_bias"))

    def getWeights(self):
        return self.weights

    def setWeights(self, data):
        self.weights = PopulatedTensor(data)
        self.weights.name = MangledName(
            OriginalName(
                self.name.stringifyOriginalName() +
                "_weights"))
        print(" -> Setting weights:", self.weights.name)

    def loadOutputChannels(self, outputChannels):
        self.outputChannels = outputChannels

    def loadTrainedParameters(self, **kwargs):
        self.weights = PopulatedTensor(kwargs['weights'], name=str(self.name) + "_weights")
        try:
            self.bias = PopulatedTensor(kwargs['bias'], name=str(self.name) + "_bias")
        except BaseException:
            #print("No Bias")
            pass

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


class Deconvolution(Convolution2DUniversal):
    def __init__(self, *args):
        super().__init__(*args)

        # Set the supported layouts
        tfCM = TensorFormat(Layouts.NHWC, (1,))
        self.formatPool = [(tfCM, tfCM)]

    def loadGroupSize(self, groupSize):
        self.groupSize = groupSize

    def convert2Conv(self):
        # See
        # http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#transposed-convolution

        if (self.strideHeight == 1 and self.strideWidth == 1 and
            self.paddingWidth == self.kernelWidth / 2 and
                self.paddingHeight == self.kernelHeight / 2):
            # Equivalent convolution has padding p' = k - p - 1
            paddingHeight = self.kernelHeight - self.paddingHeight - 1
            paddingWidth = self.kernelWidth - self.paddingWidth - 1

            conv = Convolution2D(
                self.name,
                self.inputTensorNames,
                self.outputTensorNames)
            for attr_name in self.__dict__:
                setattr(conv, attr_name, getattr(self, attr_name))

            conv.loadPadding(paddingHeight, paddingWidth)

            return {"conv": conv, "upsampling": None}
        else:
            upsampled_tensor_name = MangledName(OriginalName(
                self.getInputTensorNames()[0].stringifyOriginalName()))
            upsampling_layer = Upsampling(
                self.name.stringifyOriginalName() + "_upsampling",
                self.getInputTensorNames(),
                [upsampled_tensor_name]
            )
            upsampling_layer.setInputTensors(self.getInputTensors())
            upsampling_layer.loadInputTensorSizes(self.getInputTensorSizes())
            upsampling_layer.compute_and_set_upsampling_parameters_for_deconv(
                self.strideWidth, self.strideHeight,
                self.kernelWidth, self.kernelHeight,
                self.paddingWidth, self.paddingHeight
            )

            upsampled_tensor = UnpopulatedTensor(
                shape=upsampling_layer.getOutputTensorSizes()[0])
            upsampled_tensor.setName(upsampling_layer.getOutputTensorNames()[0])
            upsampling_layer.setOutputTensors([upsampled_tensor])

            conv = Convolution2D(
                self.name.stringifyOriginalName() + "_as_conv",
                upsampling_layer.getOutputTensorNames(),
                self.outputTensorNames,
            )

            conv.setInputTensors([upsampled_tensor])
            conv.loadInputTensorSizes(upsampling_layer.getOutputTensorSizes())
            conv.setOutputTensors(self.getOutputTensors())
            conv.loadOutputTensorSizes(self.getOutputTensorSizes())
            conv.loadPadding(0, 0)
            conv.loadStrideSize(1, 1)
            conv.loadDilation(self.dilationFactor)
            conv.setBiasEnabled(self.biasEnabled())
            conv.loadKernelSize(*self.getKernelSize())
            conv.loadGroupSize(self.groupSize)
            conv.loadOutputChannels(self.outputChannels)
            if (self.biasEnabled()):
                conv.bias = self.bias
            conv.weights = self.weights

            return {"conv": conv, "upsampling": upsampling_layer}


class Convolution2D(Convolution2DUniversal):
    def __init__(self, *args):
        super().__init__(*args)

        # Set the supported layouts
        tfCM = TensorFormat(Layouts.NHWC, ())
        self.formatPool = [(tfCM, tfCM)]

    def loadGroupSize(self, groupSize):
        self.groupSize = groupSize

    def getGroupSize(self):
        return self.groupSize

    def genGroupConvs(self):

        group_filter = (1, self.getGroupSize(), 1, 1)
        group_shape = list(map(lambda a, b: a //
                               b, self.getInputTensors()[0].getShape(), group_filter))
        group_channels = group_shape[1]

        out_ch = self.getOutputTensors()[0].getShape()[1]
        output_group_shape = list(
            map(lambda a, b: a // b, self.getOutputTensors()[0].getShape(), group_filter))

        group_convs = []

        for group in range(self.getGroupSize()):

            # Take our slice of Weights
            group_N_weights = self.weights.data[self.weights.data.shape[0] // self.getGroupSize(
            ) * group: self.weights.data.shape[0] // self.getGroupSize() * (group + 1), ]

            # Our slice of Input...
            group_N_input = UnpopulatedTensor(group_shape)
            group_N_input.setName(
                MangledName(
                    OriginalName(
                        self.getInputTensors()[0].name.stringifyOriginalName())))

            # And Output...
            outputSlice = UnpopulatedTensor((output_group_shape))
            outputSlice.setName(
                MangledName(
                    OriginalName(
                        self.getOutputTensors()[0].name.stringifyOriginalName())))

            # Create the new Convolution representhing this group.
            new_name = self.name.stringifyOriginalName()
            l = Convolution2D(new_name, None, None)

            l.setInputTensorsAllFields([group_N_input])
            l.setOutputTensorsAllFields([outputSlice])

            # Set group size
            l.loadGroupSize(1)
            l.sliced = (group, self.getGroupSize())

            # Apply our split of Weights
            l.setWeights(group_N_weights)

            # Copy some of the other properties of the original convolution.
            (kh, kw) = self.getKernelSize()
            (sh, sw) = self.getStrideSize()
            (ph, pw) = self.getPadding()
            l.loadKernelSize(kh, kw)
            l.loadStrideSize(sh, sw)
            l.loadPadding(ph, pw)
            l.loadDilation(self.getDilation())

            if self.biasEnabled():
                default_qp = QuantizationParameters(np.array([1]), np.array([0]))
                bias_data = self.getBias().data.flatten()
                l.setBias(bias_data[group *
                                    (out_ch //
                                     self.getGroupSize()): (group +
                                                            1) *
                                    (out_ch //
                                        self.getGroupSize())])  # Correct?
                l.setBiasEnabled(True)
                l.getBias().quantization = default_qp
            else:
                l.setBiasEnabled(False)

            c = Crop("crop_" + l.name.stringifyOriginalName(), [], [])
            c.setInputTensorsAllFields(self.getInputTensors())
            c.setOutputTensorsAllFields([group_N_input])
            c.setOffset(np.array([self.getInputTensors()[
                        0].shape[1] // self.getGroupSize() * group, 0, 0]))
            c.setImplicit()

            group_convs.append([l, c])

        return group_convs


class ConvolutionDepthWise2D(Convolution2DUniversal):
    def __init__(self, *args):
        super().__init__(*args)
        # TODO: Merge layout representations
        self.addCompatibleLayout(Layouts.NCHW)    # Planar
        self.addCompatibleLayout(Layouts.NHCW)    # Row Interleaved

        # Set the supported layouts
        tfCM = TensorFormat(Layouts.NHWC, (1,))
        tfIV = TensorFormat(Layouts.NHCW, (1, 2, 3), axesAlignment=(1, 1, 1, 8))

        # In mobilenet ssd, priorbox forces convertion layers. If type 'any' of TensorFormat
        # is implemented, this should not be necessary
        from Controllers.Globals import USING_MA2480
        if USING_MA2480:
            self.formatPool = [(tfIV, tfIV), (tfCM, tfCM)]
        else:
            self.formatPool = [(tfCM, tfCM), (tfIV, tfIV)]

    def loadGroupSize(self, groupSize):
        self.groupSize = groupSize

    def getGroupSize(self):
        return self.groupSize

    def convert2GroupConv(self, scheduler):

        group_conv = Convolution2D(
            'group_' + self.name.stringifyOriginalName(), [], [])

        # Copy attributes
        for attr_name in self.__dict__:
            setattr(group_conv, attr_name, getattr(self, attr_name))

        implementAsSwLayer, new_group_size = self.getNewGroupSize(scheduler)
        if implementAsSwLayer:
            return self

        new_weights = np.zeros(
            (self.weights.data.shape[0],
             self.groupSize *
             self.weights.data.shape[1] //
             new_group_size,
             self.kernelHeight,
             self.kernelWidth))

        stepk = self.weights.shape[0] // self.groupSize
        stepc = self.weights.shape[1]
        max_c = new_weights.shape[1]
        for idx, group in enumerate(range(self.groupSize)):
            group_N_weights = self.weights.data[self.weights.data.shape[0] //
                                                self.groupSize *
                                                group: self.weights.data.shape[0] //
                                                self.groupSize *
                                                (group +
                                                 1), ]
            new_weights[stepk *
                        idx:stepk *
                        (idx +
                         1), stepc *
                        (idx %
                         max_c):stepc *
                        ((idx %
                          max_c) +
                            1), ...] = group_N_weights

        group_conv.setWeights(new_weights)
        group_conv.groupSize = new_group_size

        return group_conv

    def getNewGroupSize(self, scheduler):

        i = self.getInputTensors()[0].shape
        o = self.getOutputTensors()[0].shape
        # Ensure compatibility with old code
        stage = {'kx': self.kernelWidth, 'ky': self.kernelHeight,
                 'sx': self.strideWidth, 'sy': self.strideHeight,
                 'ic': i[1], 'oc': o[1],
                 'is': [i[3], i[2]],
                 'os': [o[3], o[2]],
                 'pkx': self.kernelWidth, 'pky': self.kernelHeight,
                 'pkx': self.strideWidth, 'pky': self.strideHeight}
        sw, newGroups = scheduler.optimize_depthwise_convolution(stage=stage)

        return sw, newGroups
