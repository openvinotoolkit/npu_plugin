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
from ctypes import *
from Models.EnumDeclarations import *
from Controllers.HwDescriptorSerializer import SerializedDescriptor, HwDescOp


class HwDescriptorList:
    def __init__(self):
        self.descList = []
    def setInputAddress(self, baseInputAddress):
        for desc in self.descList:
            desc.adjustInputAddress(baseInputAddress)
    def setOutputAddress(self, baseOutputAddress):
        for desc in self.descList:
            desc.adjustOutputAddress(baseOutputAddress)
    def setTapsAddress(self, baseTapsAddress):
        for desc in self.descList:
            desc.adjustTapsAddress(baseTapsAddress)
    def setBiasAddress(self, baseBiasAddress):
        for desc in self.descList:
            desc.adjustBiasAddress(baseBiasAddress)
    def setScaleAddress(self, baseScaleAddress):
        for desc in self.descList:
            desc.adjustScaleAddress(baseScaleAddress)
    def pushDescriptor(self, descriptor):
        self.descList.append(descriptor)
    def getContentSize(self):
        return 16 * 8 * len(self.descList)
    def getContent(self, baseAddress):
        content = []
        relocInstance = []
        relocWorkBuffer = []
        relocInBlob = []

        for desc in self.descList:
            (dc, dri, drwb, drib), lastDescAddress = desc.getContent(baseAddress)
            content.extend(dc)
            relocInstance.extend(dri)
            relocWorkBuffer.extend(drwb)
            relocInBlob.extend(drib)
        return content, relocInstance, relocWorkBuffer, relocInBlob, lastDescAddress


class HwDescriptor:
    def __init__(self, dataMode, id, disableInt, interruptTrigger, mode, tileIndex, lastTile, stageName):
        self.dataMode = dataMode
        self.opType = HwDescOp.convolution.value
        self.id = id
        self.disableInt = disableInt
        self.disableInt = 0
        self.interruptTrigger = interruptTrigger
        self.tileIndex = tileIndex
        self.interleavedInput = 0
        self.interleavedOutput = 0
        self.sodGroup = 0
        self.sohGroup = 0
        self.actualOutChannels = 0
        self.topOutputJunk = 0
        self.bottomOutputJunk = 0
        self.interleaved = False
        self.lastTile = lastTile
        self.inputDataAddr = 0
        self.inputDimX = 0
        self.inputDimY = 0
        self.inputDimZ = 0
        self.outputDataAddr = 0
        self.outputDimX = 0
        self.outputDimY = 0
        self.outputDimZ = 0
        self.coeffMode = 0
        self.coeffData = 0
        self.kerDimX = 0
        self.kerDimY = 0
        self.stride = 0
        self.poolEn = 0
        self.poolType = 0
        self.poolKerDimX = 0
        self.poolKerDimY = 0
        self.accumulate = 0
        self.totalDimX = 0
        self.vectorData = 0
        self.noOfVectors = 0
        self.padEn = 0
        self.padMode = 0
        self.reluEn = 0
        self.reluXEn = 0
        self.t0 = 0
        self.a0 = 0
        self.a1 = 0
        self.x = 0
        self.reuseData = 0
        self.biasAddress = 0
        self.scaleAddress = 0
        self.useBias = False
        self.useScale = False
        self.mode = mode
        self.type = 0
        self.stageName = stageName
    def adjustInputAddress(self, baseInputAddr):
        self.inputDataAddr += baseInputAddr
    def adjustOutputAddress(self, baseOutputAddr):
        self.outputDataAddr += baseOutputAddr
    def adjustTapsAddress(self, baseTapsAddr):
        self.coeffData += baseTapsAddr
    def adjustBiasAddress(self, baseBiasAddr):
        if self.useBias:
            self.biasAddress += baseBiasAddr
    def adjustScaleAddress(self, baseScaleAddr):
        if self.useScale:
            self.scaleAddress += baseScaleAddr
    def setupInput(self, inputDataAddr, inputDimX, inputDimY, inputDimZ, totalInputDimY, totalInputDimZ):
        self.type = 0
        self.inputDataAddr = inputDataAddr
        self.inputDimX = inputDimX
        self.inputDimY = inputDimY
        self.inputDimZ = inputDimZ
        self.totalInputDimY = totalInputDimY
        self.totalInputDimZ = totalInputDimZ
    def setupInterleaved(self, interleaved):
        self.interleaved = interleaved
    def setupOutput(self, outputDataAddr, outputDimX, outputDimY, outputDimZ, totalOutputDimY, totalOutputDimZ):
        self.type = 0
        self.outputDataAddr = outputDataAddr
        self.outputDimX = outputDimX
        self.outputDimY = outputDimY
        self.outputDimZ = outputDimZ
        self.totalOutputDimY = totalOutputDimY
        self.totalOutputDimZ = totalOutputDimZ
    def setupConvolutionCoefficients(self, mode, coeffData, kerDimX, kerDimY, stride):
        self.type = 0
        self.coeffMode = mode
        self.coeffData = coeffData
        self.kerDimX = kerDimX
        self.kerDimY = kerDimY
        self.stride = stride
        self.type = 0
    def setupConvolutionPooling(self, poolType, poolKerDimX, poolKerDimY):
        self.type = 0
        self.opType = HwDescOp.convolution_with_pooling.value
        self.poolType = poolType
        self.poolEn = 1
        self.poolKerDimX = poolKerDimX
        self.poolKerDimY = poolKerDimX
    def setupPooling(self, poolType, poolKerDimX, poolKerDimY, stride):
        self.type = 4
        self.poolType = poolType
        self.poolEn = 1
        self.poolKerDimX = poolKerDimX
        self.poolKerDimY = poolKerDimY
        self.stride = stride
    def setupInputFullyConnected(self, inputDataAddr, accumulate, totalDimX, inputDimX):
        self.type = 2
        self.inputDataAddr = inputDataAddr
        self.accumulate = accumulate
        self.totalDimX = totalDimX
        self.inputDimX = inputDimX
    def setupOutputFullyConnected(self, outputDataAddr):
        self.type = 2
        self.outputDataAddr = outputDataAddr
    def setupVectorsFullyConnected(self, coeffMode, vectorData, noOfVectors):
        self.type = 2
        self.coeffMode = coeffMode
        self.vectorData = vectorData
        self.noOfVectors = noOfVectors
    def setupPadding(self, padMode):
        self.padEn = 1
        self.padMode = padMode #0
    def setupRelu(self, t0, a0, a1):
        self.reluEn = 1
        self.t0 = t0
        self.a0 = a0
        self.a1 = a1
    def setupReluX(self, x):
        self.reluXEn = 1
        self.x = x
    def setupBias(self, bias):
        self.useBias = True
        self.biasAddress = bias
    def setupScale(self, scale):
        self.useScale = True
        self.scaleAddress = scale
    def setupInterleavedInput(self, interleaved):
        self.interleavedInput = interleaved
    def setupInterleavedOutput(self, interleaved):
        self.interleavedOutput = interleaved
    def getContent(self, baseAddress):
        baseAddress += self.tileIndex * 16 * 8
        if self.type == 0:
            return self.getContentForConvolution(baseAddress), baseAddress
        elif self.type == 4:
            return self.getContentForPooling(baseAddress), baseAddress
        elif self.type == 2:
            return self.getContentForFullyConnected(baseAddress), baseAddress
        return None
    def getContentForConvolution(self, baseAddress, debug=False):
        noOfBlocks = 1 << self.mode
        sizeOfBlock = (128 * 1024) >> self.mode
        bytesPerPixel = 1 << (1 - self.dataMode)
        pixelsPerCMXLine = 128 // (bytesPerPixel * 8)
        inDataLineStride = bytesPerPixel * self.inputDimX
        inDataLineStride = ((inDataLineStride + 15) // 16) * 16
        inDataChanStride = inDataLineStride * self.totalInputDimY

        if self.interleavedInput:
            # Assume interleaved input:
            inDataChanStride = inDataLineStride
            inDataLineStride = inDataLineStride * self.totalInputDimZ
        
        localLineStride = (self.inputDimX + (pixelsPerCMXLine - 1)) // pixelsPerCMXLine

        chanPerBlock = self.inputDimZ // noOfBlocks
        availableBytesPerChan = sizeOfBlock // chanPerBlock
        bytesPerLine = localLineStride * pixelsPerCMXLine * bytesPerPixel
        linesPerChan = availableBytesPerChan // bytesPerLine
        if(linesPerChan > self.inputDimY):
            linesPerChan = self.inputDimY
        localChanStride = linesPerChan * localLineStride
        if(self.poolEn == 1):
            minLines = self.kerDimY + self.poolKerDimY
        else:
            minLines = min(self.kerDimY + 1, linesPerChan)
        coeffLPB = chanPerBlock * self.kerDimY * self.kerDimX
        coeffSetSize = self.kerDimX * self.kerDimY
        outDataLineStride = bytesPerPixel * self.outputDimX
        outDataLineStride = ((outDataLineStride + 15) // 16) * 16
        outDataChanStride = outDataLineStride * self.totalOutputDimY

        if self.interleavedOutput:
            # Assume interleaved output:
            outDataChanStride = outDataLineStride
            outDataLineStride = outDataLineStride * self.totalOutputDimZ

        bytesPerCoeffSet = coeffSetSize
        coeffChStrideIn = bytesPerCoeffSet * 2 * 8
        coeffChStrideOut = coeffChStrideIn * self.inputDimZ
        content = []
        relocInstance = []
        relocWorkBuffer = [baseAddress + 8*4, baseAddress + 16*4]
        relocInBlob = [baseAddress + 12*4]

        #Line 0
        nextDescAddr = 0
        if not self.lastTile:
            nextDescAddr = baseAddress + (16 * 8)
            relocInstance = [baseAddress + 0]

        if self.useBias:
            relocInBlob.append(baseAddress + 22*4)
        if self.useScale:
            relocInBlob.append(baseAddress + 23*4)

        chemicalX = 0
        if(self.reluXEn):
            chemicalX = self.x
        elif(self.poolEn and self.poolType == 1):
            chemicalX = np.float16(1 / (self.poolKerDimX * self.poolKerDimY)).view(np.uint16)


        # If not set, ensure fields default to 0, not -1
        if self.poolKerDimX == 0:
            self.poolKerDimX = 1
        if self.poolKerDimY == 0:
            self.poolKerDimY = 1

        assert(minLines <= linesPerChan)

        sd = SerializedDescriptor("Conv")

        # Line 0
        sd.set_field("NextDesc", nextDescAddr)
        sd.set_field("Type", self.opType)    # TODO:
        sd.set_field("mode", self.mode)
        sd.set_field("rsvd_00", self.interleavedInput + (self.interleavedOutput << 1))
        sd.set_field("id", self.id)
        sd.set_field("it", self.interruptTrigger)
        sd.set_field("cm", 0)
        sd.set_field("dm", 0)
        sd.set_field("disaint", self.disableInt)
        sd.set_field("rsvd_02", 0)

        # Line 1
        sd.set_field("iDimY-1", self.inputDimY - 1 )
        sd.set_field("rsvd_10", self.topOutputJunk )
        sd.set_field("iDimX-1", self.inputDimX - 1 )
        sd.set_field("rsvd_11", self.bottomOutputJunk)
        sd.set_field("iChans-1", self.inputDimZ - 1)
        sd.set_field("rsvd_12", 0)
        sd.set_field("oChans-1", self.outputDimZ - 1)
        sd.set_field("interleaved", self.interleaved)

        # Line 2
        sd.set_field("ChRamBlk-1", chanPerBlock-1)
        sd.set_field("stride", self.stride-1)
        sd.set_field("InFw-1", self.kerDimX-1)
        sd.set_field("InFh-1", self.kerDimY-1)
        sd.set_field("PadType", self.padMode)
        sd.set_field("PadEnable", self.padEn)

        # Line 3
        sd.set_field("poolEn", self.poolEn)
        sd.set_field("poolKernelHeight-1", self.poolKerDimX-1)
        sd.set_field("poolKernelWidth-1", self.poolKerDimY-1)
        sd.set_field("avgPoolX", chemicalX)
        sd.set_field("poolType", self.poolType)

        # Line 4
        sd.set_field("dataBaseAddr", self.inputDataAddr)
        sd.set_field("t0", self.t0)
        sd.set_field("a0", self.a0)
        sd.set_field("a1", self.a1)
        sd.set_field("reluXEn", self.reluXEn)
        sd.set_field("reluEn", self.reluEn)

        # Line 5
        sd.set_field("dataChStr", inDataChanStride)
        sd.set_field("dataLnStr", inDataLineStride)

        # Line 6
        sd.set_field("coeffBaseAddr", self.coeffData)
        sd.set_field("coeffChStrOut", coeffChStrideOut)

        # Line 7
        sd.set_field("coeffChStrIn", coeffChStrideIn)
        sd.set_field("outLnStr", outDataLineStride)

        # Line 8
        sd.set_field("outBaseAddr", self.outputDataAddr)
        sd.set_field("outChStr", outDataChanStride)

        # Line 9
        sd.set_field("localLs", localLineStride)
        sd.set_field("localCs", localChanStride)
        sd.set_field("linesPerCh-1", linesPerChan - 1)
        sd.set_field("rsvd_92", self.sodGroup)
        sd.set_field("rud", self.reuseData)

        # Line A
        sd.set_field("minLines-1", minLines - 1)
        sd.set_field("rsvd_A0", self.sohGroup)
        sd.set_field("coeffLpb-1", coeffLPB - 1)
        sd.set_field("css-1", coeffSetSize- 1)
        sd.set_field("outputX", self.outputDimX)

        # Line B
        sd.set_field("biasBaseAddr", self.biasAddress)
        sd.set_field("scaleBaseAddr", self.scaleAddress)

        sd.set_pallete(None)


        content = sd.serialize()

        return content, relocInstance, relocWorkBuffer, relocInBlob

    def getContentForPooling(self, baseAddress, debug=False):
        noOfBlocks = 1 << self.mode
        sizeOfBlock = (128 * 1024) >> self.mode
        bytesPerPixel = 1 << (1 - self.dataMode)
        pixelsPerCMXLine = 128 // (bytesPerPixel * 8)
        inDataLineStride = bytesPerPixel * self.inputDimX
        inDataLineStride = ((inDataLineStride + 15) // 16) * 16
        inDataChanStride = inDataLineStride * self.totalInputDimY

        if self.interleavedInput:
            # Assume interleaved input:
            inDataChanStride = inDataLineStride
            inDataLineStride = inDataLineStride * self.totalInputDimZ

        localLineStride = (self.inputDimX + (pixelsPerCMXLine - 1)) // pixelsPerCMXLine
        chanPerBlock = self.inputDimZ // noOfBlocks
        availableBytesPerChan = sizeOfBlock // chanPerBlock
        bytesPerLine = localLineStride * pixelsPerCMXLine * bytesPerPixel
        linesPerChan = availableBytesPerChan // bytesPerLine
        if(linesPerChan > self.inputDimY):
            linesPerChan = self.inputDimY
        localChanStride = linesPerChan * localLineStride
        # if(self.poolEn == 1):
        #     minLines = self.kerDimY + self.poolKerDimY
        # else:
        #     minLines = self.kerDimY + 1

        if self.inputDimX / self.stride <= 4:
            minLines = self.poolKerDimY + 2*self.stride + 3
        else:
            minLines = self.kerDimY + 1
            minLines = self.poolKerDimY + self.stride + 2

        outDataLineStride = bytesPerPixel * self.outputDimX
        outDataLineStride = ((outDataLineStride + 15) // 16) * 16
        outDataChanStride = outDataLineStride * self.outputDimY

        if self.interleavedOutput:
            # Assume interleaved output:
            outDataChanStride = outDataLineStride
            outDataLineStride = outDataLineStride * self.totalOutputDimZ

        content = []
        relocInstance = []
        relocWorkBuffer = [baseAddress + 8*4, baseAddress + 16*4]
        relocInBlob = [baseAddress + 12*4]



        if self.useBias:
            relocInBlob.append(baseAddress + 22*4)
        if self.useScale:
            relocInBlob.append(baseAddress + 23*4)

        chemicalX = 0
        if(self.reluXEn):
            chemicalX = self.x
        elif(self.poolEn and self.poolType == 1):
            chemicalX = np.float16(1 / (self.poolKerDimX * self.poolKerDimY)).view(np.uint16)


        #Line 0
        nextDescAddr = 0
        if not self.lastTile:
            nextDescAddr = baseAddress + (16 * 8)
            relocInstance = [baseAddress + 0]

        # If not set, ensure fields default to 0, not -1
        if self.poolKerDimX == 0:
            self.poolKerDimX = 1
        if self.poolKerDimY == 0:
            self.poolKerDimY = 1
        if self.kerDimX == 0:
            self.kerDimX = 1
        if self.kerDimY == 0:
            self.kerDimY = 1

        sd = SerializedDescriptor("Pool")

        # Line 0
        sd.set_field("NextDesc", nextDescAddr)
        sd.set_field("Type", HwDescOp.pooling_only.value)
        sd.set_field("mode", self.mode)
        sd.set_field("rsvd_00", self.interleavedInput + (self.interleavedOutput << 1))
        sd.set_field("id", self.id)
        sd.set_field("it", self.interruptTrigger)
        sd.set_field("cm", 0)
        sd.set_field("dm", 0)
        sd.set_field("disaint", self.disableInt)
        sd.set_field("rsvd_02", 0)

        # Line 1
        sd.set_field("iDimY-1", self.inputDimY - 1 )
        sd.set_field("rsvd_10", self.topOutputJunk )
        sd.set_field("iDimX-1", self.inputDimX - 1 )
        sd.set_field("rsvd_11", self.bottomOutputJunk)
        sd.set_field("iChans-1", self.inputDimZ - 1)
        sd.set_field("oChans-1", self.outputDimZ - 1)
        sd.set_field("interleaved", self.interleaved)

        # Line 2
        sd.set_field("ChRamBlk-1", chanPerBlock-1)
        sd.set_field("stride", self.stride-1)
        sd.set_field("InFw-1", self.kerDimX-1)
        sd.set_field("InFh-1", self.kerDimY-1)
        sd.set_field("PadType", self.padMode)
        sd.set_field("PadEnable", self.padEn)

        # Line 3
        sd.set_field("poolEn", self.poolEn)
        sd.set_field("poolKernelHeight-1", self.poolKerDimX-1)
        sd.set_field("poolKernelWidth-1", self.poolKerDimY-1)
        sd.set_field("avgPoolX", chemicalX)
        sd.set_field("poolType", self.poolType)

        # Line 4
        sd.set_field("dataBaseAddr", self.inputDataAddr)
        sd.set_field("t0", self.t0)
        sd.set_field("a0", self.a0)
        sd.set_field("a1", self.a1)
        sd.set_field("reluXEn", self.reluXEn)
        sd.set_field("reluEn", self.reluEn)

        # Line 5
        sd.set_field("dataChStr", inDataChanStride)
        sd.set_field("dataLnStr", inDataLineStride)

        # Line 6
        # Nothing needed Here

        # Line 7
        sd.set_field("outLnStr", outDataLineStride)

        # Line 8
        sd.set_field("outBaseAddr", self.outputDataAddr)
        sd.set_field("outChStr", outDataChanStride)

        # Line 9
        sd.set_field("localLs", localLineStride)
        sd.set_field("localCs", localChanStride)
        sd.set_field("linesPerCh-1", linesPerChan - 1)
        sd.set_field("rsvd_92", self.sodGroup)
        sd.set_field("rud", self.reuseData)

        # Line A
        sd.set_field("minLines-1", minLines - 1)
        sd.set_field("rsvd_A0", self.sohGroup)
        sd.set_field("outputX", self.outputDimX)

        # Line B
        sd.set_field("biasBaseAddr", self.biasAddress)
        sd.set_field("scaleBaseAddr", self.scaleAddress)

        sd.set_pallete(None)

        content = sd.serialize()

        return content, relocInstance, relocWorkBuffer, relocInBlob

    def getContentForFullyConnected(self, baseAddress):
        noOfBlocks = 1 << self.mode
        bytesPerCoefficient = 1 << (1 - self.dataMode)
        pixelsPerBlock = self.inputDimX // noOfBlocks
        inDataLineStride = 16
        inDataBlockStride = inDataLineStride * pixelsPerBlock
        outDataLineStride = 16
        outDataBlockStride = outDataLineStride * pixelsPerBlock
        localLineStride = 16
        localBlockStride = localLineStride * pixelsPerBlock
        vectorLPB = self.inputDimX // noOfBlocks
        vectStrideIn = bytesPerCoefficient * pixelsPerBlock * 8
        vectStrideOut = bytesPerCoefficient * self.totalDimX * 8
        content = []
        relocInstance = []
        relocWorkBuffer = [baseAddress + 8*4, baseAddress + 16*4]
        relocInBlob = [baseAddress + 12*4]

        nextDescAddr = 0
        if not self.lastTile:
            nextDescAddr = baseAddress + (16 * 8)
            relocInstance = [baseAddress + 0]

        # If not set, ensure fields default to 0, not -1

        if self.poolKerDimX <= 0:
            self.poolKerDimX = 1
        if self.poolKerDimY <= 0:
            self.poolKerDimY = 1

        if self.useBias:
            relocInBlob.append(baseAddress + 22*4)
        if self.useScale:
            relocInBlob.append(baseAddress + 23*4)

        sd = SerializedDescriptor("FCL")

        # Line 0
        sd.set_field("NextDesc", nextDescAddr)
        sd.set_field("Type", HwDescOp.fully_connected_convolution.value)
        sd.set_field("mode", self.mode)
        sd.set_field("rsvd_00", self.interleavedInput + (self.interleavedOutput << 1))
        sd.set_field("id", self.id)
        sd.set_field("it", self.interruptTrigger)
        sd.set_field("cm", 0)
        sd.set_field("dm", 0)
        sd.set_field("disaint", self.disableInt)
        sd.set_field("rsvd_02", 0)

        # Line 1
        sd.set_field("iDimX-1", self.inputDimX - 1 )
        sd.set_field("iChans-1", self.noOfVectors - 1)
        sd.set_field("oChans-1", self.noOfVectors - 1)

        # Line 2
        sd.set_field("ChRamBlk-1", pixelsPerBlock-1)

        # Line 3
        sd.set_field("actualOutChannels", self.actualOutChannels-1)
        sd.set_field("X", self.x)

        # Line 4
        sd.set_field("dataBaseAddr", self.inputDataAddr)
        sd.set_field("t0", self.t0)
        sd.set_field("a0", self.a0)
        sd.set_field("a1", self.a1)
        sd.set_field("reluXEn", self.reluXEn)
        sd.set_field("reluEn", self.reluEn)

        # Line 5
        sd.set_field("dataChStr", inDataBlockStride)
        sd.set_field("dataLnStr", inDataLineStride)

        # Line 6
        sd.set_field("vecStrOut", vectStrideOut)
        sd.set_field("vecBaseAddr", self.vectorData)

        # Line 7
        sd.set_field("vecStrIn", vectStrideIn)
        sd.set_field("outLnStr", outDataLineStride)

        # Line 8
        sd.set_field("outBaseAddr", self.outputDataAddr)
        sd.set_field("outChStr", outDataBlockStride)

        # Line 9
        sd.set_field("localLs", localLineStride)
        sd.set_field("localBs", localBlockStride)
        sd.set_field("rud", self.reuseData)

        # Line A
        sd.set_field("Acc", self.accumulate)
        sd.set_field("vecLPB-1", vectorLPB-1)
        sd.set_field("outputX", 1)

        # Line B
        sd.set_field("biasBaseAddr", self.biasAddress)
        sd.set_field("scaleBaseAddr", self.scaleAddress)

        sd.set_pallete(None)

        content = sd.serialize()

        return content, relocInstance, relocWorkBuffer, relocInBlob

