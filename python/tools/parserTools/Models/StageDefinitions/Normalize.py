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


class Normalize(Op):

    def __init__(self):
        super().__init__("Normalize")

    def specific_details_push(self, target_container, instance):
        helper_parseBuffer("input", target_container, instance.dataBUF)
        helper_parseBuffer("output", target_container, instance.outputBUF)
        helper_parseBuffer("taps", target_container, instance.tapsBUF)
        helper_parseBuffer("op_parmas", target_container, instance.opParamsBUF)

    def adapt_fields(self, emulator, layer):
        from Controllers.Adaptor import BufferEmulator
        from Controllers.Tensor import PopulatedTensor

        in_tensor = layer.getInputTensors()[0]
        in_tensor.setDatatype(np.float16)
        emulator.dataBUF = BufferEmulator(in_tensor.resolve())

        out_tensor = layer.getOutputTensors()[0]
        out_tensor.setDatatype(np.float16)
        emulator.outputBUF = BufferEmulator(out_tensor.resolve())

        w = layer.scales
        w.setDatatype(np.float16)
        k, c, kh, kw = w.getShape()
        w.reshape((1, k, c, kh * kw))
        emulator.tapsBUF = BufferEmulator(w.resolve())

        normalize_params = np.array((layer.across_spatial,
                           layer.channel_shared,
                           layer.epsilon),
                           dtype = np.dtype("<i4, <i4, <f4")).flatten().view(np.int32)
        opParamsTensor = PopulatedTensor(normalize_params)
        opParamsTensor.setLayout((0, 1, 2, 3))
        opParamsTensor.setDatatype(np.int32)
        emulator.opParamsBUF = BufferEmulator(opParamsTensor.resolve(), track=True)
