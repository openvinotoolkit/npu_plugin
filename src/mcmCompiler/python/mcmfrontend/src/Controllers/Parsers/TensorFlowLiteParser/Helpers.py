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

from Controllers.Parsers.TensorFlowLiteParser.tflite.TensorType import TensorType
from Controllers.Parsers.TensorFlowLiteParser.tflite.Padding import Padding
from Controllers.Parsers.TensorFlowLiteParser.tflite.ActivationFunctionType import ActivationFunctionType as actType
from Controllers.Parsers.Parser.ReLU import ReLU
from Controllers.Parsers.Parser.tan_h import TanH

import numpy as np


def getTensorIndices(obj):
    in_tensors = [obj.Inputs(idx) for idx in range(obj.InputsLength())]
    out_tensors = [obj.Outputs(idx) for idx in range(obj.OutputsLength())]

    return in_tensors, out_tensors


def getTensorType(tensor):
    tensor_type = tensor.Type()
    if tensor_type == TensorType.FLOAT32:
        return np.float32
    elif tensor_type == TensorType.FLOAT16:
        return np.float16
    elif tensor_type == TensorType.UINT8:
        return np.uint8
    elif tensor_type == TensorType.INT32:
        return np.int32
    else:
        raise ValueError("Format {} is not supported!".format(tensor_type))


def setTensorShape(layer, input_tensors, output_tensors):
    def __get_shape_list(tensors):
        shapes = []
        for t in tensors:
            if t.ShapeLength() == 2:
                shapes.append([t.Shape(0), t.Shape(1), 1, 1])
            else:
                shapes.append([t.Shape(0), t.Shape(3), t.Shape(1), t.Shape(2)])
        return shapes

    layer.loadInputTensorSizes(__get_shape_list(input_tensors))
    layer.loadOutputTensorSizes(__get_shape_list(output_tensors))


def getActivation(options, out):
    if options.FusedActivationFunction() == actType.RELU:
        act = ReLU(out.Name().decode("utf-8") +
                   "/ReLU", [out.Name()], [out.Name()])
    elif options.FusedActivationFunction() == actType.TANH:
        act = TanH(out.Name().decode("utf-8") +
                   "/TanH", [out.Name()], [out.Name()])
    elif options.FusedActivationFunction() == actType.NONE:
        return None
    else:
        raise ValueError(
            "Unsupported activation function {}".format(
                options.FusedActivationFunction()))

    setTensorShape(act, [out], [out])
    return act


def decodePadding(padding):
    if padding == Padding.VALID:
        return 'VALID'
    elif padding == Padding.SAME:
        return 'SAME'
    else:
        raise ValueError("Invalid Padding")


def getPadding(in_dim, kernel_dim, stride_dim, padding_type):
    def same_padding(in_dim, kernel_dim, stride_dim):
        import numpy as np
        """
        Calculates the output dimension and also the padding required for that dimension.
        :param in_dim: Width/Height of Input
        :param kernel_dim: Width/Height of Kernel
        :param stride_dim: Vertical/Horizontal Stride
        """
        in_dim = np.array(in_dim)
        in_dim[in_dim is None] = 1

        kernel_dim = np.array(kernel_dim)
        stride_dim = np.array(stride_dim)

        output_dim = np.ceil(np.float_(in_dim) / np.float_(stride_dim))
        pad = ((output_dim - 1) * stride_dim + kernel_dim - in_dim)
        # account for negative pad cases
        pad = np.clip(pad, 0, np.amax(pad))
        pad = [(int(pad[idx] // 2), int(pad[idx] - pad[idx] // 2))
               for idx in range(len(pad))]
        return pad

    def valid_padding(in_dim, kernel_dim, stride_dim):
        # output_dim = np.ceil(np.float_(in_dim - kernel_dim + 1) / np.float_(stride_dim))
        pad = [(0, 0)] * len(in_dim)  # as many zeros as there are dimensions

        return pad

    if padding_type == Padding.VALID:
        return valid_padding(in_dim, kernel_dim, stride_dim)
    elif padding_type == Padding.SAME:
        return same_padding(in_dim, kernel_dim, stride_dim)
    else:
        return None


def stripTensorName(tensorName):
    if tensorName[-2:] == ':0':
        return tensorName[0:-2]
    else:
        return tensorName


# find tensor value
def findTensorValue(tensor, buffers):
    idx = tensor.Buffer()
    # Read the buffer (little endian)
    data_buffer = buffers[idx].DataAsNumpy()
    dt = getTensorType(tensor)
    if dt == np.float32:
        data = np.frombuffer(data_buffer, dtype=dt).astype(np.float16)
    else:
        data = np.frombuffer(data_buffer, dtype=dt)

    data = np.reshape(data, [tensor.Shape(idx)
                             for idx in range(tensor.ShapeLength())])

    return data
