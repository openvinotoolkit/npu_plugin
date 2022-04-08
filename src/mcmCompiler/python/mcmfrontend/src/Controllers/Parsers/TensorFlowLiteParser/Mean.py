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
from Controllers.Parsers.Parser.Pooling import Pooling
from Controllers.Parsers.TensorFlowLiteParser.tflite.BuiltinOperator import BuiltinOperator

from Controllers.Parsers.TensorFlowLiteParser.tflite.MeanOptions import MeanOptions

from .Helpers import getPadding, findTensorValue, getTensorIndices, decodePadding, setTensorShape, getActivation


def load(obj, op_type, tensors, buffers):

    in_tensors, out_tensors = getTensorIndices(obj)

    assert(len(in_tensors) == 2)  # input and axis
    assert(len(out_tensors) == 1)  # output

    activation = tensors[in_tensors[0]]
    axis = tensors[in_tensors[1]]

    out = tensors[out_tensors[0]]
    # operation only supports one input Tensor and output Tensor

    axis_data = findTensorValue(axis, buffers)

    # Mean operation currently only supports 2D Mean
    assert(len(axis_data) == 2)

    x = Pooling(out.Name(), [activation.Name()], [out.Name()])
    mean = MeanOptions()

    x.loadType(Pooling.Type.AVE)

    mean.Init(obj.BuiltinOptions().Bytes, obj.BuiltinOptions().Pos)

    # TODO: Utilize keep_dims option
    keep_dims = mean.KeepDims()

    # Strides are always 1 for Mean
    stride_w = 1
    stride_h = 1
    x.loadStrideSize(stride_h, stride_w)

    # The kernel is implied from axis indices
    kernelSize = (activation.Shape(axis_data[0]), activation.Shape(axis_data[1]))
    x.loadKernelSize(kernelSize[0], kernelSize[1])

    # Padding is always ((0, 0), (0, 0)) for Mean
    padding = ((0, 0), (0, 0))
    x.loadPadding(padding[0], padding[1])
    x.setPadStyle("SAME")

    # Set tensors shapes
    setTensorShape(x, [activation], [out])

    x.loadGlobal(kernelSize == (activation.Shape(1), activation.Shape(2)))

    return [x]
