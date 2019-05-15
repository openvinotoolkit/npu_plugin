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
from Models.EnumDeclarations import *
from Models.StageDefinitions.Op import *
from Models.Blob import helper_parseBuffer
from Controllers.BlobBuilder import *
from Controllers.PingPong import PingPongSchedule, getManualHwSchedule, get_null_terminating_name

class MyriadXHardwareLayer(Op):

    def __init__(self):
        super().__init__("MyriadXHardwareLayer")

        self.requirements = {
            "input": {
                "layout": StorageOrder.orderZYX
            },
            "weights":{
                "layout": TapsOrder.orderKCHW
            },
            "Hardware": True
        }

    def specific_details_push(self, target_container, instance):

        input_stride, output_stride, newDimZ, BufSize, hwDescList, taps, _, _ = instance.hardware_solution
        content, relocInst, relocWork, relocBlob, lastDesc = hwDescList.getContent(0)

        if isinstance(instance.name, str):
            stageName = instance.name
            pingPongPair = instance.pingPongPair
        else:
            stageName = get_null_terminating_name(instance.name)
            pingPongPair = getManualHwSchedule()
        dataBUFinCMX = 0
        outputBUFinCMX = 0
        streaming = 0


        # Force FC not to get streamed
        if pingPongPair.isStreamed(stageName) and \
            instance.op not in [StageType.myriadX_fully_connected_layer]:
            streaming = 1

        streamingCmxPos = pingPongPair.streamingCmxPos(stageName)
        assert(streamingCmxPos in ['L', 'R'])

        streamingCmxPos = 0 if streamingCmxPos == 'L' else 1

        cmxSizeForStreaming = pingPongPair.cmxForStreaming(stageName)

        if pingPongPair.sourceInCmxA(stageName):
            dataBUFinCMX = 1
        elif pingPongPair.sourceInCmxB(stageName):
            dataBUFinCMX = 2
        elif pingPongPair.sourceInCmxAuxA(stageName):
            dataBUFinCMX = 3
        elif pingPongPair.sourceInCmxAuxB(stageName):
            dataBUFinCMX = 4

        if pingPongPair.destinationInCmxA(stageName):
            outputBUFinCMX = 1
        elif pingPongPair.destinationInCmxB(stageName):
            outputBUFinCMX = 2
        elif pingPongPair.destinationInCmxAuxA(stageName):
            outputBUFinCMX = 3
        elif pingPongPair.destinationInCmxAuxB(stageName):
            outputBUFinCMX = 4

        unloadCmxBuffer = 0
        if pingPongPair.unloadCmxBuffer(stageName):
            unloadCmxBuffer = 1

        overwriteInput = 0
        if pingPongPair.overwriteInput(stageName):
            overwriteInput = 1

        concatOffset = newDimZ[2]
        reluOnShaveAccumulation = newDimZ[3]
        shvNegSlope = newDimZ[4]
        shvPosSlope = newDimZ[5]

        inputBufferSize, outputBufferSize = BufSize[0:2]
        debug = False


        target_container.push("HwDesc_op_mode", Value(c_uint32(streaming | (streamingCmxPos << 2) | (dataBUFinCMX << 4) | (outputBUFinCMX << 8) )))
        target_container.push("HwDesc_inputSize", Value(c_uint32(inputBufferSize)))
        target_container.push("HwDesc_outputSize", Value(c_uint32(outputBufferSize)))
        target_container.push("HwDesc_concatOffset", Value(c_uint32(concatOffset)))
        target_container.push("HwDesc_unloadCmxBuffer", Value(c_uint32(unloadCmxBuffer)))
        target_container.push("HwDesc_overwriteInput", Value(c_uint32(overwriteInput)))
        target_container.push("HwDesc_cmxSizeForStreaming", Value(c_int32(cmxSizeForStreaming)))
        target_container.push("HwDesc_reluOnShaveAccumulation", Value(c_uint32(reluOnShaveAccumulation)))
        target_container.push("HwDesc_shvNegSlope", Value(c_float(shvNegSlope)))
        target_container.push("HwDesc_shvPosSlope", Value(c_float(shvPosSlope)))
        target_container.push("HwDesc_descr_num", Value(c_uint32(len(hwDescList.descList))))
        for x in range(len(content) // 32):
            for y in range(32):
                target_container.push("HwDesc Half-Line "+str(x)+":"+str(y), Value(content[x*32+y]))
                if debug:
                    print("HwDesc Half-Line "+str(x)+"|"+str(y)+ ":    ", format(content[x*32+y].value, '08x'))
            if debug:
                print("")

        helper_parseBuffer("input", target_container, instance.dataBUF)
        helper_parseBuffer("output", target_container, instance.outputBUF)
        helper_parseBuffer("taps", target_container, instance.tapsBUF)
        helper_parseBuffer("bias", target_container, instance.biasBUF)
        helper_parseBuffer("scale", target_container, instance.scaleBUF)

    def adapt_fields(self, emulator, layer):

        from Controllers.Adaptor import BufferEmulator  # TODO: Fix Imports.
        from Controllers.Tensor import PopulatedTensor  # TODO: Fix Imports.

        layer.compile()
        emulator.name = layer.getName().stringifyOriginalName()
        # Handle Ping Pong Crap
        emulator.pingPongPair = PingPongSchedule({emulator.name: layer.getPingPong()})

        # TODO: Data should be correctly populated by calling code.
        i = layer.getInputTensors()[0]
        i.setDatatype(np.float16)
        emulator.dataBUF    = BufferEmulator(i.resolve())

        o = layer.getOutputTensors()[0]
        o.setDatatype(np.float16)
        emulator.outputBUF  = BufferEmulator(o.resolve())

        if layer.hasWeights():

            w = layer.getWeights()
            w.setDatatype(np.float16)
            k, c, kh, kw = w.getShape()
            w.reshape((1, kh * kw, c, k))
            w.setLayout((0, 1, 2, 3))
            emulator.tapsBUF    = BufferEmulator(w.resolve())


            if layer.biasEnabled():
                b = layer.getBias()
                b.setLayout((0, 1, 2, 3))
                b.setDatatype(np.float16)
                b.shape = (0, 0, 0, 0)
                emulator.biasBUF    = BufferEmulator(b.resolve())
            else:
                emulator.biasBUF    = BufferEmulator(None)

            # # TODO: Implement Scale
            if layer.scaleEnabled():
                scale = layer.scale * layer.relu_scale
                scaleBUF = PopulatedTensor(scale * np.ones((k)))
                scaleBUF.setDatatype(np.float16)
                scaleBUF.shape = (0, 0, 0, 0)
                scaleBUF.setLayout((0, 2, 3, 1))
                emulator.scaleBUF   = BufferEmulator(scaleBUF.resolve())
            else:
                emulator.scaleBUF   = BufferEmulator(None)
        else:

            emulator.tapsBUF    = BufferEmulator(None)
            emulator.biasBUF    = BufferEmulator(None)
            emulator.scaleBUF   = BufferEmulator(None)

        emulator.hardware_solution = layer.hardware_solution

