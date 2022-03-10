//
// Copyright 2022 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#ifdef CONFIG_HAS_LRT_SRCS
#include <nn_log.h>
#else
#define nnLog(level, ...)
#endif
#include <mvSubspaces.h> // for CHAR_BIT
#include <param_gather.h>

using namespace sw_params; // for GatherParams
using namespace subspace; // for getCoords getOffsetU8

extern "C" {
void singleShaveGather(uint32_t lParams) {
    const GatherParams* layerParams = reinterpret_cast<const GatherParams*>(lParams);

    half* pActInput = (half*)(layerParams->input.dataAddr);
    int32_t* pActIndices = (int32_t*)(layerParams->indices.dataAddr);
    int32_t axis = *(int32_t*)(layerParams->axis.dataAddr);
    half* pActOutput = (half*)(layerParams->output.dataAddr);

    int32_t numInputDims = (int32_t)layerParams->input.numDims;
    int32_t* pInputDims = (int32_t*)layerParams->input.dimsAddr;
    int64_t* pInputStrides = (int64_t*)layerParams->input.stridesAddr;

    int32_t numIndicesDims = (int32_t)layerParams->indices.numDims;
    int32_t* pIndicesDims = (int32_t*)layerParams->indices.dimsAddr;
    int64_t* pIndicesStrides = (int64_t*)layerParams->indices.stridesAddr;

    int32_t numOutputDims = (int32_t)layerParams->output.numDims;
    int32_t* pOutputDims = (int32_t*)layerParams->output.dimsAddr;
    int64_t* pOutputStrides = (int64_t*)layerParams->output.stridesAddr;

    // TODO: Check window in kernel
    half* pActWindowFp16 = (half*)(layerParams->windowfp16.dataAddr);
    half* pActWindowFp16Values = pActWindowFp16;
    for (int i = 0; i < 4 * 3 * 2; i++) {
        *pActWindowFp16Values = *pActInput;
        if (i == 0) {
            *pActWindowFp16Values = 1234;
        }
        pActWindowFp16Values++;
        pActInput++;
    }

    int32_t* pActWindowInt32 = (int32_t*)(layerParams->windowint32.dataAddr);
    int32_t* pActWindowInt32Values = pActWindowInt32;
    for (int i = 0; i < 4 * 3 * 2; i++) {
        *pActWindowInt32Values = 1;
        if (i == 0) {
            *pActWindowInt32Values = axis;
        }
        pActWindowInt32Values++;
    }

    pActWindowInt32Values = pActWindowInt32 + 1;
    for (int i = 1; i < 2 * 3 + 1; i++) {
        *pActWindowInt32Values = *pActIndices;
        pActWindowInt32Values++;
        pActIndices++;
    }

    pActWindowInt32[7] = numInputDims; // 3
    pActWindowInt32[8] = pInputDims[0]; // 4
    pActWindowInt32[9] = pInputDims[1]; // 3
    pActWindowInt32[10] = pInputDims[2]; // 2
    pActWindowInt32[11] = pInputDims[3]; // -1
    pActWindowInt32[12] = pInputStrides[0] / CHAR_BIT; // 2
    pActWindowInt32[13] = pInputStrides[1] / CHAR_BIT; // 8
    pActWindowInt32[14] = pInputStrides[2] / CHAR_BIT; // 24
    pActWindowInt32[15] = pInputStrides[3] / CHAR_BIT; // 0

    pActWindowInt32[16] = numIndicesDims; // 2
    pActWindowInt32[17] = pIndicesDims[0]; // 2
    pActWindowInt32[18] = pIndicesDims[1]; // 3
    pActWindowInt32[19] = pIndicesDims[2]; // -1
    pActWindowInt32[20] = pIndicesDims[3]; // -1
    pActWindowInt32[21] = pIndicesStrides[0] / CHAR_BIT; // 4
    pActWindowInt32[22] = pIndicesStrides[1] / CHAR_BIT; // 8
    pActWindowInt32[23] = pIndicesStrides[2] / CHAR_BIT; // 0
    pActWindowInt32[24] = pIndicesStrides[3] / CHAR_BIT; // 0

    pActWindowInt32[25] = numOutputDims; // 4
    pActWindowInt32[26] = pOutputDims[0]; // 4
    pActWindowInt32[27] = pOutputDims[1]; // 2
    pActWindowInt32[28] = pOutputDims[2]; // 3
    pActWindowInt32[29] = pOutputDims[3]; // 2
    pActWindowInt32[30] = pOutputStrides[0] / CHAR_BIT; // 2
    pActWindowInt32[31] = pOutputStrides[1] / CHAR_BIT; // 8
    pActWindowInt32[32] = pOutputStrides[2] / CHAR_BIT; // 16
    pActWindowInt32[33] = pOutputStrides[3] / CHAR_BIT; // 48

    // TODO: Check window in kernel End

    int numOutputValues = getTotal(pOutputDims, numOutputDims); // 48
    pActWindowInt32[34] = numOutputValues;

    int32_t inputCoords[4] = {0, 0, 0, 0};
    int32_t indicesCoords[4] = {0, 0, 0, 0};
    int32_t outputCoords[4] = {0, 0, 0, 0};

    int32_t indicesStrides[4] = {0, 0, 0, 0};
    for(int i = 0; i < numIndicesDims; i++) {
        indicesStrides[i] = pIndicesStrides[i] / CHAR_BIT;
    }

    for (int outIdx = 0; outIdx < numOutputValues; outIdx++) {
        // get output coords
        getCoord(outIdx, pOutputDims, numOutputDims, outputCoords); // 5 [1][1][0][0]

        // generate indices coords and offset
        for(int axisIdx = 0; axisIdx < numIndicesDims; axisIdx++) {
            indicesCoords[axisIdx] = outputCoords[axis + axisIdx];
        }
        int indicesOffset = getOffsetU8(indicesCoords, indicesStrides, numIndicesDims); // 20 for [1][2] 4+2*8

        // get indices value as input index
        const uint8_t* pIndicesValue = (uint8_t*)pActIndices + indicesOffset;
        int inputDimIdx = *(int32_t*)pIndicesValue;

        // generate input coords and offset
        for(int inIdx = 0; inIdx < numInputDims; inIdx++) {
            if (inIdx < axis) {
                inputCoords[inIdx] = outputCoords[inIdx];
            } else if (inIdx == axis) {
                inputCoords[inIdx] = inputDimIdx;
            }
            else {
                inputCoords[inIdx] = outputCoords[inIdx + numIndicesDims - 1];
            }
        }

        // get input value as output value

        // set output value by output offset

    }

}
}
