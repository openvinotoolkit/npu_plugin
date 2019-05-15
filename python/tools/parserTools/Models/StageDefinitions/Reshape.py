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


class Reshape(Op):

    def __init__(self):
        super().__init__("Reshape")

    def specific_details_push(self, target_container, instance):

        target_container.push("offset_dimX",  Value(c_uint32(int(instance.outputDimX))))
        target_container.push("offset_dimY",  Value(c_uint32(int(instance.outputDimY))))
        target_container.push("offset_dimZ",  Value(c_uint32(int(instance.outputDimZ))))

        helper_parseBuffer("input", target_container, instance.dataBUF)
        helper_parseBuffer("output", target_container, instance.outputBUF)

    def adapt_fields(self, emulator, layer):

        from Controllers.Adaptor import BufferEmulator  # TODO: Fix Imports.
        from Controllers.Tensor import PopulatedTensor  # TODO: Fix Imports.


        # TODO: Data should be correctly populated by calling code.
        i = layer.getInputTensors()[0]
        i.setLayout((0, 2, 3, 1))
        i.setDatatype(np.float16)
        emulator.dataBUF    = BufferEmulator(i.resolve())

        o = layer.getOutputTensors()[0]
        output_shape = o.getShape()
        assert(len(output_shape) == 4)
        emulator.outputDimX = output_shape[3]
        emulator.outputDimY = output_shape[2]
        emulator.outputDimZ = output_shape[1]
        o.setLayout((0, 2, 3, 1))
        o.setDatatype(np.float16)
        emulator.outputBUF  = BufferEmulator(o.resolve())
