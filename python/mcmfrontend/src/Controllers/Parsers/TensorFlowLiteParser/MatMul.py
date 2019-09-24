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
from Controllers.Parsers.Parser.InnerProduct import InnerProduct
from Controllers.Parsers.TensorFlowLiteParser.tflite.BuiltinOperator import BuiltinOperator

from Controllers.Parsers.TensorFlowLiteParser.tflite.FullyConnectedOptions import FullyConnectedOptions

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

    x = InnerProduct(out.Name(), [activation.Name()], [out.Name()])
    options = FullyConnectedOptions()

    if out.ShapeLength() == 2:
        outputChannels = out.Shape(1)
    else:
        outputChannels = out.Shape(3)
    x.loadOutputChannels(outputChannels)

    options.Init(obj.BuiltinOptions().Bytes, obj.BuiltinOptions().Pos)

    weight_data = findTensorValue(weight, buffers)
    if len(weight_data.shape) == 2:
        weight_data = np.expand_dims(
            np.expand_dims(
                np.transpose(
                    weight_data, (1, 0)), -1), 0)
    else:
        weight_data = np.transpose(weight_data, (3, 2, 0, 1))

    bias_data = findTensorValue(bias, buffers)
    x.setBiasEnabled(True)
    x.loadTrainedParameters(weights=weight_data, bias=bias_data)
    x.setQuantizationParameters(weight.Quantization(), bias.Quantization())

    setTensorShape(x, [activation], [out])

    fused_activation = getActivation(options, out)

    if fused_activation:
        return [x, fused_activation]
    else:
        return [x]
