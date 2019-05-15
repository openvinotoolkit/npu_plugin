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


class AveragePooling(Op):

    def __init__(self):
        super().__init__("AveragePooling")

    def specific_details_push(self, target_container, instance):
        target_container.push("radixX", Value(c_uint32(instance.radixX)))
        target_container.push("radixY", Value(c_uint32(instance.radixY)))
        target_container.push("radixStrideX", Value(c_uint32(instance.strideX)))
        target_container.push("radixStrideY", Value(c_uint32(instance.strideY)))
        target_container.push("padX", Value(c_uint32(instance.padX)))
        target_container.push("padY", Value(c_uint32(instance.padY)))
        target_container.push("padStyle", Value(c_uint32(instance.padStyle.value)))
        helper_parseBuffer("input", target_container, instance.dataBUF)
        helper_parseBuffer("output", target_container, instance.outputBUF)

    def adapt_fields(self, emulator, layer):

        from Controllers.Adaptor import BufferEmulator  # TODO: Fix Imports.

        emulator.radixX = layer.kernelWidth
        emulator.radixY = layer.kernelHeight
        emulator.strideX = layer.strideWidth
        emulator.strideY = layer.strideHeight
        emulator.padX = layer.paddingWidth
        emulator.padY = layer.paddingHeight
        emulator.padStyle = layer.padStyle

        # TODO: Does Pooling care about opParams, bias or taps Buffer

        i = layer.getInputTensors()[0]
        i.setDatatype(np.float16)
        emulator.dataBUF    = BufferEmulator(i.resolve())


        o = layer.getOutputTensors()[0]
        o.setDatatype(np.float16)
        emulator.outputBUF  = BufferEmulator(o.resolve())

        emulator.tapsBUF    = BufferEmulator(None)
        emulator.biasBUF    = BufferEmulator(None)