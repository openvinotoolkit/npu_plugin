from ctypes import *

from Models.StageDefinitions.Op import *
from Models.Blob import helper_parseBuffer
from Controllers.BlobBuilder import *


class EluOp(Op):

    def __init__(self):
        super().__init__("Elu")

    def specific_details_push(self, target_container, instance):
        target_container.push("opX",  Value(c_uint32(int(instance.post_param1))))
        target_container.push("post_strideX",  Value(c_uint32(instance.post_strideX)))
        target_container.push("post_strideY",  Value(c_uint32(instance.post_strideY)))
        helper_parseBuffer("input", target_container, instance.dataBUF)
        helper_parseBuffer("output", target_container, instance.outputBUF)

    def adapt_fields(self, emulator, layer):

        from Controllers.Adaptor import BufferEmulator  # TODO: Fix Imports.

        emulator.post_param1 = layer.alpha
        emulator.post_strideX = 0
        emulator.post_strideY = 0

        i = layer.getInputTensors()[0]
        i.setDatatype(np.float16)
        emulator.dataBUF = BufferEmulator(i.resolve())

        o = layer.getOutputTensors()[0]
        o.setDatatype(np.float16)
        emulator.outputBUF = BufferEmulator(o.resolve())
