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
from Controllers.Parsers.Parser.ReLU import ReLU, LeakyReLU
from Controllers.Parsers.Parser.PReLU import PReLU
from Controllers.Parsers.TensorFlowLiteParser.tflite.BuiltinOperator import BuiltinOperator
from Controllers.Parsers.TensorFlowLiteParser.tflite.LeakyReluOptions import LeakyReluOptions


from .Helpers import getPadding, findTensorValue, getTensorIndices, decodePadding, setTensorShape, getActivation


def load(obj, op_type, tensors, buffers):

    in_tensors, out_tensors = getTensorIndices(obj)

    assert(len(in_tensors) == 1)  # input
    assert(len(out_tensors) == 1)  # output

    activation = tensors[in_tensors[0]]
    out = tensors[out_tensors[0]]

    if op_type in [BuiltinOperator.RELU, BuiltinOperator.RELU6]:
        x = ReLU(out.Name(), [activation.Name()], [out.Name()])
    elif op_type in [BuiltinOperator.LEAKY_RELU, BuiltinOperator.PRELU]:
        # The difference between the two is only at training stage...
        x = LeakyReLU(out.Name(), [activation.Name()], [out.Name()])
        options = LeakyReluOptions()
        options.Init(obj.BuiltinOptions().Bytes, obj.BuiltinOptions().Pos)
        x.loadAlpha(options.Alpha())
    else:
        raise ValueError("Layer {} not supported".format(op_type))

    setTensorShape(x, [activation], [out])

    return [x]
