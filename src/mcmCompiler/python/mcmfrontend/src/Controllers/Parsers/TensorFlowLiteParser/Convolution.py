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

import numpy as np
from Controllers.Parsers.Parser.Convolution2D import (
    Convolution2D, ConvolutionDepthWise2D, Deconvolution)
from Controllers.Parsers.TensorFlowLiteParser.tflite.BuiltinOperator import BuiltinOperator

from Controllers.Parsers.TensorFlowLiteParser.tflite.Conv2DOptions import Conv2DOptions
from Controllers.Parsers.TensorFlowLiteParser.tflite.TransposeConvOptions import TransposeConvOptions
from Controllers.Parsers.TensorFlowLiteParser.tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions

from .Helpers import getPadding, findTensorValue, getTensorIndices, decodePadding, setTensorShape, getActivation


def load(obj, op_type, tensors, buffers):

    in_tensors, out_tensors = getTensorIndices(obj)

    assert(len(in_tensors) == 3)  # input and weights
    assert(len(out_tensors) == 1)  # output

    activation = tensors[in_tensors[0]]
    weight = tensors[in_tensors[1]]
    bias = tensors[in_tensors[2]]

    out = tensors[out_tensors[0]]
    # operation only supports one input Tensor and output Tensor

    if op_type == BuiltinOperator.CONV_2D:
        x = Convolution2D(out.Name(), [activation.Name()], [out.Name()])
        groupSize = 1  # Conv2D
        x.loadGroupSize(groupSize)
        conv = Conv2DOptions()
    elif op_type == BuiltinOperator.DEPTHWISE_CONV_2D and activation.Shape(3) == 1:
        x = Convolution2D(out.Name(), [activation.Name()], [out.Name()])
        groupSize = 1  # Conv2D
        x.loadGroupSize(groupSize)
        conv = DepthwiseConv2DOptions()
    elif op_type == BuiltinOperator.DEPTHWISE_CONV_2D:
        x = ConvolutionDepthWise2D(
            out.Name(), [
                activation.Name()], [
                out.Name()])
        conv = DepthwiseConv2DOptions()
    elif op_type == BuiltinOperator.TRANSPOSE_CONV:
        x = Deconvolution(out.Name(), [activation.Name()], [out.Name()])
        conv = TransposeConvOptions()
    else:  # else not supported layer
        raise ValueError("Layer type not supported by Convolution: " + obj.type)

    outputChannels = out.Shape(3)
    x.loadOutputChannels(outputChannels)

    conv.Init(obj.BuiltinOptions().Bytes, obj.BuiltinOptions().Pos)

    if op_type != BuiltinOperator.TRANSPOSE_CONV:
        # ensure dilation is equal
        dilation_w = conv.DilationWFactor()
        dilation_h = conv.DilationHFactor()
        if(dilation_w != dilation_h):
            raise ValueError("Different dilation in not supported")
        # dilation in batch and channels are always 1
        x.loadDilation(dilation_w)
    else:
        # Dilation is not supported for deconvolution in TFLite
        x.loadDilation(1)

    stride_w = conv.StrideW()
    stride_h = conv.StrideH()
    x.loadStrideSize(stride_h, stride_w)

    # The kernel is stored as KHWC
    kernelSize = (weight.Shape(1), weight.Shape(2))
    x.loadKernelSize(kernelSize[0], kernelSize[1])

    padding = getPadding([activation.Shape(idx) for idx in range(
        1, 3)], kernelSize, (stride_h, stride_w), conv.Padding())

    x.loadPadding(padding[0], padding[1])

    x.setPadStyle(decodePadding(conv.Padding()))

    weight_data = findTensorValue(weight, buffers)
    weight_data = np.transpose(weight_data, (0, 3, 1, 2))

    if (op_type == BuiltinOperator.DEPTHWISE_CONV_2D and \
       activation.Shape(3) == 1):
        weight_data = np.transpose(weight_data, (1, 0, 2, 3))

    bias_data = findTensorValue(bias, buffers)
    x.setBiasEnabled(True)
    x.loadTrainedParameters(weights=weight_data, bias=bias_data)
    x.setQuantizationParameters(weight.Quantization(), bias.Quantization())

    setTensorShape(x, [activation], [out])

    fused_activation = getActivation(conv, out)

    if fused_activation:
        return [x, fused_activation]
    else:
        return [x]
