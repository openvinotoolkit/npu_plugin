//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "mvSubspaces.h"
#include <param_gather.h>

using namespace sw_params;
using namespace subspace;

extern "C" {
void single_shave_gather(uint32_t lParams) {
    const GatherParams* layerParams = reinterpret_cast<const GatherParams*>(lParams);

    half* pActInput = (half*)(layerParams->input.dataAddr);
    int32_t* pActIndices = (int32_t*)(layerParams->indices.dataAddr);
    int64_t axis = layerParams->axis;
    int64_t batchDims = layerParams->batchDims;
    half* pActOutput = (half*)(layerParams->output.dataAddr);

    int32_t numInputDims = (int32_t)layerParams->input.numDims;
    int32_t* pInputDims = (int32_t*)layerParams->input.dimsAddr;
    int64_t* pInputStrides = (int64_t*)layerParams->input.stridesAddr;

    int32_t numIndicesDims = (int32_t)layerParams->indices.numDims;
    int64_t* pIndicesStrides = (int64_t*)layerParams->indices.stridesAddr;

    int32_t numOutputDims = (int32_t)layerParams->output.numDims;
    int32_t* pOutputDims = (int32_t*)layerParams->output.dimsAddr;
    int64_t* pOutputStrides = (int64_t*)layerParams->output.stridesAddr;

    int32_t inputCoords[MAX_GATHER_DIMS] = {0};
    int32_t indicesCoords[MAX_GATHER_DIMS] = {0};
    int32_t outputCoords[MAX_GATHER_DIMS] = {0};

    int32_t inputStrides[MAX_GATHER_DIMS] = {0};
    for (int i = 0; i < numInputDims; i++) {
        inputStrides[i] = pInputStrides[i] / CHAR_BIT;
    }
    int32_t indicesStrides[MAX_GATHER_DIMS] = {0};
    for (int i = 0; i < numIndicesDims; i++) {
        indicesStrides[i] = pIndicesStrides[i] / CHAR_BIT;
    }
    int32_t outputStrides[MAX_GATHER_DIMS] = {0};
    for (int i = 0; i < numOutputDims; i++) {
        outputStrides[i] = pOutputStrides[i] / CHAR_BIT;
    }

    int indicesValueBound = pInputDims[axis];
    int numOutputValues = getTotal(pOutputDims, numOutputDims);

    for (int outIdx = 0; outIdx < numOutputValues; outIdx++) {
        // get output coords
        getCoord(outIdx, pOutputDims, numOutputDims, outputCoords);

        // generate indices coords and offset
        for (int axisIdx = 0; axisIdx < numIndicesDims; axisIdx++) {
            if (axisIdx < numIndicesDims - batchDims) {
                indicesCoords[axisIdx] = outputCoords[axis + axisIdx];
            } else {
                indicesCoords[axisIdx] = outputCoords[numOutputDims - numIndicesDims + axisIdx];
            }
        }
        int indicesOffset = getOffsetU8(indicesCoords, indicesStrides, numIndicesDims);

        // get indices value as input index
        const uint8_t* pIndicesValue = (uint8_t*)pActIndices + indicesOffset;
        int inputIdxOnAxis = *(int32_t*)pIndicesValue;

        // set output to 0 if indices are out of bound
        int outputOffset = getOffsetU8(outputCoords, outputStrides, numOutputDims);
        const uint8_t* pOutputValue = (uint8_t*)pActOutput + outputOffset;
        if (inputIdxOnAxis < -indicesValueBound && inputIdxOnAxis > indicesValueBound - 1) {
            *(half*)pOutputValue = 0;
            continue;
        }

        // reverse negative indices
        if (inputIdxOnAxis < 0) {
            inputIdxOnAxis += indicesValueBound;
        }

        // generate input coords and offset
        for (int inIdx = 0; inIdx < numInputDims; inIdx++) {
            if (inIdx < axis) {
                inputCoords[inIdx] = outputCoords[inIdx];
            } else if (inIdx == axis) {
                inputCoords[inIdx] = inputIdxOnAxis;
            } else {
                inputCoords[inIdx] = outputCoords[inIdx + numIndicesDims - batchDims - 1];
            }
        }
        int inputOffset = getOffsetU8(inputCoords, inputStrides, numInputDims);

        // set output value by corresponding input value
        const uint8_t* pInputValue = (uint8_t*)pActInput + inputOffset;
        *(half*)pOutputValue = *(half*)pInputValue;
    }
}
}
