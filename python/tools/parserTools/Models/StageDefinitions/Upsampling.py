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

from ctypes import *

from Models.StageDefinitions.Op import *
from Models.Blob import helper_parseBuffer
from Controllers.BlobBuilder import *


class Upsampling(Op):

    def __init__(self):
        super().__init__("Upsampling")

    def specific_details_push(self, target_container, instance):
        target_container.push("upsampling_factor_x", Value(c_int32(instance.opParams[0])))
        target_container.push("upsampling_factor_y", Value(c_int32(instance.opParams[1])))
        target_container.push("upsampling_factor_z", Value(c_int32(instance.opParams[2])))

        target_container.push("pad_x_0", Value(c_int32(instance.opParams[3])))
        target_container.push("pad_x_1", Value(c_int32(instance.opParams[4])))

        target_container.push("pad_y_0", Value(c_int32(instance.opParams[5])))
        target_container.push("pad_y_1", Value(c_int32(instance.opParams[6])))

        target_container.push("pad_z_0", Value(c_int32(instance.opParams[7])))
        target_container.push("pad_z_1", Value(c_int32(instance.opParams[8])))

        helper_parseBuffer("input", target_container, instance.dataBUF)
        helper_parseBuffer("output", target_container, instance.outputBUF)

    def adapt_fields(self, emulator, layer):

        from Controllers.Adaptor import BufferEmulator  # TODO: Fix Imports.
        from Controllers.Tensor import PopulatedTensor

        i = layer.getInputTensors()[0]
        i.setDatatype(np.float16)
        emulator.dataBUF = BufferEmulator(i.resolve())

        o = layer.getOutputTensors()[0]
        o.setDatatype(np.float16)
        emulator.outputBUF = BufferEmulator(o.resolve())

        emulator.opParams = [
            layer.upsampling_factor_W,
            layer.upsampling_factor_H,
            layer.upsampling_factor_C,
            layer.output_pad_W[0],
            layer.output_pad_W[1],
            layer.output_pad_H[0],
            layer.output_pad_H[1],
            layer.output_pad_C[0],
            layer.output_pad_C[1]
            ]
