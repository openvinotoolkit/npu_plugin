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


class DetectionOutput(Op):

    def __init__(self):
        super().__init__("DetectionOutput")

    def specific_details_push(self, target_container, instance):
        helper_parseBuffer("input", target_container, instance.dataBUF)
        helper_parseBuffer("output", target_container, instance.outputBUF)
        helper_parseBuffer("taps", target_container, instance.tapsBUF)
        helper_parseBuffer("bias", target_container, instance.biasBUF)
        helper_parseBuffer("op_parmas", target_container, instance.opParamsBUF)

    def adapt_fields(self, emulator, layer):
        from Controllers.Adaptor import BufferEmulator  # TODO: Fix Imports.
        from Controllers.Tensor import PopulatedTensor

        i = layer.getInputTensors()[0]
        i.setDatatype(np.float16)
        emulator.dataBUF    = BufferEmulator(i.resolve())

        w = layer.getInputTensors()[1]
        w.setDatatype(np.float16)
        emulator.tapsBUF    = BufferEmulator(w.resolve())

        b = layer.getInputTensors()[2]
        b.setDatatype(np.float16)
        emulator.biasBUF    = BufferEmulator(b.resolve())

        o = layer.getOutputTensors()[0]
        o.setDatatype(np.float16)
        emulator.outputBUF  = BufferEmulator(o.resolve())

        det_out_params = layer.params
        opParamsTensor = PopulatedTensor(det_out_params)
        opParamsTensor.setLayout((0, 1, 2, 3))
        opParamsTensor.setDatatype(np.float32)
        emulator.opParamsBUF = BufferEmulator(opParamsTensor.resolve(), track=True)

