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


class Crop(Op):

    def __init__(self):
        super().__init__("Crop")

    def specific_details_push(self, target_container, instance):

        target_container.push("offset_dimX",  Value(c_uint32(int(instance.opParams[0]))))
        target_container.push("offset_dimY",  Value(c_uint32(int(instance.opParams[1]))))
        target_container.push("offset_dimZ",  Value(c_uint32(int(instance.opParams[2]))))

        helper_parseBuffer("input", target_container, instance.dataBUF)
        helper_parseBuffer("output", target_container, instance.outputBUF)

    def adapt_fields(self, emulator, layer):
        from Controllers.Adaptor import BufferEmulator  # TODO: Fix Imports.
        from Controllers.Tensor import PopulatedTensor  # TODO: Fix Imports.

        offset = layer.getOffset()

        opParamsTensor = PopulatedTensor(offset.astype(np.int32))
        opParamsTensor.setLayout((0, 1, 2, 3))
        opParamsTensor.setDatatype(np.int32)

        emulator.opParams = BufferEmulator(opParamsTensor.resolve(), track=False)

        i = layer.getInputTensors()[0]
        i.setDatatype(np.float16)
        emulator.dataBUF    = BufferEmulator(i.resolve())

        o = layer.getOutputTensors()[0]
        o.setDatatype(np.float16)
        emulator.outputBUF  = BufferEmulator(o.resolve())
