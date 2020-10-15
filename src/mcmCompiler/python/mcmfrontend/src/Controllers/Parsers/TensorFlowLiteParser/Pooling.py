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

from Controllers.Parsers.Parser.Pooling import Pooling
from Controllers.Parsers.TensorFlowLiteParser.tflite.BuiltinOperator import BuiltinOperator

from Controllers.Parsers.TensorFlowLiteParser.tflite.Pool2DOptions import Pool2DOptions

from .Helpers import getPadding, getTensorIndices, decodePadding, setTensorShape, getActivation
from Controllers.EnumController import throw_error
from Models.EnumDeclarations import ErrorTable


def load(obj, op_type, tensors, buffers):

    in_tensors, out_tensors = getTensorIndices(obj)

    assert(len(in_tensors) == 1)  # input
    assert(len(out_tensors) == 1)  # output

    activation = tensors[in_tensors[0]]

    out = tensors[out_tensors[0]]
    # operation only supports one input Tensor and output Tensor

    x = Pooling(out.Name(), [activation.Name()], [out.Name()])
    pool = Pool2DOptions()
    if op_type == BuiltinOperator.AVERAGE_POOL_2D:
        x.loadType(Pooling.Type.AVE)
    elif op_type == BuiltinOperator.MAX_POOL_2D:
        x.loadType(Pooling.Type.MAX)
    else:  # else not supported layer
        throw_error(ErrorTable.StageDetailsNotSupported, op_type)

    pool.Init(obj.BuiltinOptions().Bytes, obj.BuiltinOptions().Pos)

    # Strides
    stride_w = pool.StrideW()
    stride_h = pool.StrideH()
    x.loadStrideSize(stride_h, stride_w)

    # The kernel is stored as KCHW
    kernelSize = (pool.FilterHeight(), pool.FilterWidth())
    x.loadKernelSize(kernelSize[0], kernelSize[1])

    # Padding
    padding = getPadding([activation.Shape(idx) for idx in range(
        1, 3)], kernelSize, (stride_h, stride_w), pool.Padding())
    x.loadPadding(padding[0], padding[1])
    x.setPadStyle(decodePadding(pool.Padding()))

    # Set tensors shapes
    setTensorShape(x, [activation], [out])

    x.loadGlobal(kernelSize == (activation.Shape(1), activation.Shape(2)))

    fused_activation = getActivation(pool, out)

    if fused_activation:
        return [x, fused_activation]
    else:
        return [x]
