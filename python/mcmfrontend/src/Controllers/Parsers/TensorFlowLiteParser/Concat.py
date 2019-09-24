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
from Controllers.Parsers.Parser.Concat import Concat
from Controllers.Parsers.TensorFlowLiteParser.tflite.BuiltinOperator import BuiltinOperator

from Controllers.Parsers.TensorFlowLiteParser.tflite.ConcatenationOptions import ConcatenationOptions

from .Helpers import getPadding, findTensorValue, getTensorIndices, decodePadding, setTensorShape, getActivation


def load(obj, op_type, tensors, buffers):

    in_tensors, out_tensors = getTensorIndices(obj)

    assert(len(in_tensors) > 1)  # input1 and input2
    assert(len(out_tensors) == 1)  # output

    activations = [tensors[in_tensor] for in_tensor in in_tensors]

    out = tensors[out_tensors[0]]

    x = Concat(out.Name(), [activation.Name()
                            for activation in activations], [out.Name()])

    options = ConcatenationOptions()
    options.Init(obj.BuiltinOptions().Bytes, obj.BuiltinOptions().Pos)

    tf2myriad_map = [0, 2, 3, 1]
    x.loadConcatAxis(tf2myriad_map[options.Axis()])

    setTensorShape(x, activations, [out])

    fused_activation = getActivation(options, out)

    if fused_activation:
        return [x, fused_activation]
    else:
        return [x]
