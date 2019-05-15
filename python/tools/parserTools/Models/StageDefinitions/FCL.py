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


class FCL(Op):

    def __init__(self):
        super().__init__("FullyConnectedLayer")

    def specific_details_push(self, target_container, instance):
        helper_parseBuffer("input", target_container, instance.dataBUF)
        helper_parseBuffer("output", target_container, instance.outputBUF)
        helper_parseBuffer("taps", target_container, instance.tapsBUF)
        helper_parseBuffer("bias", target_container, instance.biasBUF)

    def adapt_fields(self, emulator, layer):

        from Controllers.Adaptor import BufferEmulator  # TODO: Fix Imports.
        from Controllers.Tensor import PopulatedTensor  # TODO: Fix Imports.

        # TODO: Data should be correctly populated by calling code.
        i = layer.getInputTensors()[0]
        i.setDatatype(np.float16)

        _, c, height, width = i.shape
        i.reshape((1, np.prod(i.shape), 1, 1))
        emulator.dataBUF    = BufferEmulator(i.resolve())

        o = layer.getOutputTensors()[0]
        o.setDatatype(np.float16)

        # o.reorder((0, 3, 2, 1))    # Swap Width and Channels for MvTensor Dependency

        emulator.outputBUF  = BufferEmulator(o.resolve())

        w = layer.getWeights()
        w.setDatatype(np.float16)

        w.reshape((1, o.shape[1], c, height * width))
        emulator.tapsBUF    = BufferEmulator(w.resolve())

        if layer.biasEnabled():
            b = layer.getBias()
            b.setDatatype(np.float16)
            emulator.biasBUF    = BufferEmulator(b.resolve())
        else:
            emulator.biasBUF    = BufferEmulator(None)

def kchw_to_hwck(data, k=0, c=0, fh=0, fw=0):
    """
    Needed for Convolutions converted to YXZ format.
    Assumes non-flattened data. If it is flattened, pass in appropiate parameters
    :param data:
    :param k:
    :param c:
    :param fh:
    :param fw:
    :return:
    """
    k = data.shape[0] if k == 0 else k
    c = data.shape[1] if c == 0 else c
    fh = data.shape[2] if fh == 0 else fh
    fw = data.shape[3] if fw == 0 else fw
    data = data.reshape((k, c, fh, fw))

    data = np.swapaxes(data, 0, 2)  # kchw -> hckw
    data = np.swapaxes(data, 1, 3)  # hckw -> hwkc
    data = np.swapaxes(data, 2, 3)  # hckw -> hwkc

    return data
