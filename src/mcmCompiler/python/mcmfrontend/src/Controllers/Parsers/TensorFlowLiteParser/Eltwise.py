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

from Controllers.Parsers.Parser.Eltwise import Eltwise
from Controllers.Parsers.TensorFlowLiteParser.tflite.BuiltinOperator import BuiltinOperator

from Controllers.Parsers.TensorFlowLiteParser.tflite.MulOptions import MulOptions
from Controllers.Parsers.TensorFlowLiteParser.tflite.SubOptions import SubOptions
from Controllers.Parsers.TensorFlowLiteParser.tflite.AddOptions import AddOptions

from .Helpers import getTensorIndices, setTensorShape, getActivation
from Controllers.EnumController import throw_error
from Models.EnumDeclarations import ErrorTable


def load(obj, op_type, tensors, buffers):

    in_tensors, out_tensors = getTensorIndices(obj)

    assert(len(in_tensors) > 1)  # input1 and input2
    assert(len(out_tensors) == 1)  # output

    activation1 = tensors[in_tensors[0]]
    activation2 = tensors[in_tensors[1]]

    out = tensors[out_tensors[0]]
    # operation only supports one input Tensor and output Tensor

    x = Eltwise(
        out.Name(), [
            activation1.Name(), activation2.Name()], [
            out.Name()])

    if op_type in [BuiltinOperator.ADD, BuiltinOperator.SUM]:
        options = AddOptions()
        x.loadType(Eltwise.Type.WSUM)
        x.loadCoefficients([1] * len(in_tensors))
    elif op_type == BuiltinOperator.MUL:
        options = MulOptions()
        x.loadType(Eltwise.Type.WPROD)
        x.loadCoefficients([1] * len(in_tensors))
    elif op_type == BuiltinOperator.SUB:
        options = SubOptions()
        x.loadType(Eltwise.Type.WSUM)
        x.loadCoefficients([1] + [-1] * (len(in_tensors) - 1))
    else:  # else not supported layer
        throw_error(ErrorTable.StageDetailsNotSupported, op_type)

    options.Init(obj.BuiltinOptions().Bytes, obj.BuiltinOptions().Pos)

    setTensorShape(x, [activation1, activation2], [out])

    fused_activation = getActivation(options, out)

    if fused_activation:
        return [x, fused_activation]
    else:
        return [x]
