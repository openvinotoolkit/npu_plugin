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
from Controllers.Parsers.Parser.Layer import MangledName, OriginalName
from Controllers.Tensor import PopulatedTensor
from Controllers.Parsers.Parser.ReLU import ReLU
from Models import Layouts


def min_max_to_QP(Dmin, Dmax, dtype=np.uint8):
    # Takes in a min and max, and outputs a Quantization object,
    # converting the params to "scale" and "zeropoint"

    if np.uint8 and Dmin >= 0:
        scale = (Dmax - Dmin) / 255  # 2^bits -1
        scale = scale.flatten()
        qp = QuantizationParameters(np.array(scale), np.array([0]))
        qp.dtype = np.uint8
        return qp
    else:
        print("Tensorflow does not support Signed Values")
        raise NotImplementedError


class QuantizationParameters():
    def __init__(self, scale, zero_point, dtype=np.int8,
                 float_range=(-np.inf, np.inf)):
        self.scale = scale
        self.zero_point = zero_point
        self.dtype = dtype
        self.float_range = float_range

        quantization_shape = np.asarray(scale).shape
        self.shift = np.zeros(quantization_shape)
        self.mult = np.ones(quantization_shape)

    def getScale(self):
        return self.scale

    def getZeroPoint(self, output_channel=None):
        if output_channel is None or np.isscalar(self.zero_point):
            return self.zero_point
        elif self.zero_point.size == 1:
            return self.zero_point[0]
        elif self.zero_point.size > output_channel:
            return self.zero_point[output_channel]
        else:
            raise ValueError(
                "Mismatch between zero point size {} and channel {}".format(
                    self.zero_point.size, output_channel))

    def isQuantized(self):
        if self.dtype in [np.float16, np.float32, np.float64]:
            return False
        return True

    def extend(self, oc):
        if self.mult.size == 1:
            self.scale = np.ndarray([self.scale[0]] * oc)

        if self.shift.size == 1:
            self.shift = np.ndarray([self.shift[0]] * oc)

        if self.zero_point.size == 1:
            self.zero_point = np.ndarray([self.zero_point[0]] * oc)

    def float2int(self, f):
        i1 = np.round(f / self.scale + self.zero_point)
        i = i1.astype(self.dtype)
        if not self.valid_float_range(f):
            raise ValueError(
                "Conversion of {} from float to int ({}) out of safety ranges {}".format(
                    i, f, self.float_range))
        return i

    def int2float(self, q):

        f = float(q - self.zero_point) * self.scale
        if not self.valid_float_range(f):
            raise ValueError(
                "Conversion of {} to float ({}) out of safety ranges {}".format(
                    q, f, self.float_range))
        return f

    def valid_float_range(self, f):
        return f >= min(self.float_range) and f <= max(self.float_range)

    def print(self):
        # Pretty Print Quant Info
        print("=== Quantization Details ===")
        print("Zero Point: {}".format(self.zero_point))
        print("Scale: {}".format(self.scale))
        print("============================")


def quantizeLayer(layer, bits=15):
    # Quantization for Gemmlowp output
    # S1 = weight scale
    # S2 = input activation scale
    # S3 = output activation scale
    # m  = (S1 * S2)/S3, scale for MAC output
    # zeroPointScaled = output zero point scaled to MAC output precision
    # biasScaled = bias scaled to MAC output precision
    output_channels = layer.getOutputTensors()[0].shape[1]

    def extend_to_K(q):
        if np.isscalar(q):
            return q * np.ones((output_channels, 1, 1, 1)).astype(q.dtype)
        if q.size == 1:
            return np.tile(q, (output_channels, 1, 1, 1))
        elif q.size != output_channels:
            raise ValueError(
                "Quantization parameter size is not 1 or nOutchannel (size: {})".format(
                    q.size))
        else:
            return q

    if not hasattr(
            layer.getInputTensors()[0],
            'quantization') or not hasattr(
            layer.getOutputTensors()[0],
            'quantization'):
        return

    if not layer.getInputTensors()[0].quantization.isQuantized(
    ) or not layer.getOutputTensors()[0].quantization.isQuantized():
        return

    S2 = extend_to_K(layer.getInputTensors()[0].quantization.getScale())
    S3 = extend_to_K(layer.getOutputTensors()[0].quantization.getScale())

    from Controllers.Parsers.Parser.Hw import HwEltwise
    if layer.hasWeights():
        S1 = extend_to_K(layer.getWeights().quantization.getScale())
        macScale = S1 * S2
    elif isinstance(layer, HwEltwise):
        assert((layer.getInputTensors()[0].quantization.getScale(
        ) == layer.getInputTensors()[1].quantization.getScale()).all())
        macScale = S2
    else:
        macScale = S2

    # Fuse ReLU into quantization (i.e. make ReLU == saturation)
    postOps = layer.getPostOp()

    if len(postOps) == 1 and isinstance(postOps[0], ReLU) and \
            np.min(layer.getOutputTensors()[0].quantization.float_range) > 0:
        # remove post ops
        layer.post_op = []

    m = macScale / S3
    # Far better routine using frexp(m) = mantissa, exp
    mantissa, exp = np.frexp(m)
    shift = bits - exp
    mScaled = (mantissa * np.power(2, bits)).astype(np.uint16)

    shift = np.uint8(shift)
    scale = np.uint16(mScaled)
    layer.getOutputTensors()[0].quantization.shift = shift.flatten()
    layer.getOutputTensors()[0].quantization.mult = scale.flatten()

    if hasattr(layer, 'bias'):
        S_bias = layer.getBias().quantization.getScale()
        Z_bias = layer.getBias().quantization.getZeroPoint()
        # This should be true by definition but check just in case....
        assert (np.abs(S_bias - S1 * S2) / S_bias < 0.01).all() or S_bias == 0  # 1% tolerance
        assert(Z_bias == 0)

        bias = (layer.getBias().data).astype(np.int32)
        layer.setBias(bias)
    else:
        bias = PopulatedTensor(extend_to_K(np.int32(0)))
        bias.name = MangledName(
            OriginalName(
                layer.name.stringifyOriginalName() +
                "_bias"))
        layer.setBias(bias.data)

    # Hw Elementwise does not have Weights table
    from Controllers.Parsers.Parser.Hw import HwEltwise
    if isinstance(layer, HwEltwise):
        return

    # per channel layout:
    # 3 -> bias
    # 2 -> mult << 16 | round << 14 |  shift << 8 | prelu
    # 1 -> SP_PTR
    # 0 -> DATA_PTR

    if layer.bias.data.shape[-1] > 1:
        # TODO: Ensure broadcasted values
        layer.bias.data = layer.bias.data[:, :, :, 0].reshape((output_channels, 1, 1, 1))

    weights_table_data = np.zeros((output_channels, 1, 1, 4)).astype(np.int32)
    weights_table_data[:, :, :, 3:] = layer.bias.data
    scale32 = np.uint32(scale)
    shift32 = np.uint32(shift)

    # 0 - round down
    # 1 - round up
    # 2 - LFSR round
    # 3 - no round
    round_mode = 1
    round32 = np.ones(shift32.shape, np.uint32) * round_mode

    weights_table_data[:, :, :, 2:3] = (
        scale32 << 16) | (round32 << 14) | (shift32 << 8)
    layer.weights_table = PopulatedTensor(weights_table_data)
    layer.weights_table.name = MangledName(
        OriginalName(
            layer.name.stringifyOriginalName() +
            "_weights_table"))
    layer.weights_table.layout = Layouts.WeightsTableLayout
