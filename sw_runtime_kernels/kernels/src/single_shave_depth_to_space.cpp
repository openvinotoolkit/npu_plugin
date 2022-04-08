//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mvSubspaces.h>
#include <param_depth_to_space.h>

#define ND_NCHW_REV 0x4321 // reverse code of ND_NCHW 0x1234
#define ND_NHWC_REV 0x2431 // reverse code of ND_NHWC 0x1342

using namespace sw_params;
using namespace subspace;

extern "C" {
void single_shave_depth_to_space(uint32_t lParams) {
    const DepthToSpaceParams* layerParams = reinterpret_cast<const DepthToSpaceParams*>(lParams);

    half* pActInput = (half*)(layerParams->input.dataAddr);
    half* pActOutput = (half*)(layerParams->output.dataAddr);
    int32_t blockSize = (int32_t)layerParams->blockSize;
    int32_t mode = (int32_t)layerParams->mode;

    int32_t numInputDims = (int32_t)layerParams->input.numDims;
    int32_t* pInputDims = (int32_t*)layerParams->input.dimsAddr;
    NDOrder inputOrder = layerParams->input.dimsOrder;

    int32_t k = numInputDims - 2;
    int32_t inputReshapeDims[MAX_DTS_DIMS] = {0};
    int32_t orders[MAX_DTS_DIMS] = {0};

    bool isNCHWOrder = inputOrder == ND_NCHW || inputOrder == ND_NCHW_REV;
    bool isNHWCOrder = inputOrder == ND_NHWC || inputOrder == ND_NHWC_REV;

    // mode blocks_first: 0, depth_first: 1
    if (isNCHWOrder && mode == 0) {
        for (int i = 0; i < k; i++) {
            inputReshapeDims[i] = pInputDims[i];
            inputReshapeDims[i + k + 1] = blockSize;
        }
        inputReshapeDims[k] = pInputDims[k] / (blockSize * blockSize);
        inputReshapeDims[2 * k + 1] = pInputDims[k + 1];

        for (int i = 0; i < k; i++) {
            orders[2 * i] = k + 1 + i; // 2 * k + 1 - (k - i)
            orders[2 * i + 1] = i; // 2 * k + 1 - ((k - i) + k + 1)
        }
        orders[2 * k] = k;
        orders[2 * k + 1] = 2 * k + 1;
    } else if (isNCHWOrder && mode == 1) {
        for (int i = 0; i < k; i++) {
            inputReshapeDims[i] = pInputDims[i];
            inputReshapeDims[i + k] = blockSize;
        }
        inputReshapeDims[2 * k] = pInputDims[k] / (blockSize * blockSize);
        inputReshapeDims[2 * k + 1] = pInputDims[k + 1];

        for (int i = 0; i < k; i++) {
            orders[2 * i] = k + i; // 2 * k + 1 - (k + 1 - i)
            orders[2 * i + 1] = i; // 2 * k + 1 - ((k - i) + k + 1)
        }
        orders[2 * k] = 2 * k;
        orders[2 * k + 1] = 2 * k + 1;
    } else if (isNHWCOrder && mode == 0) {
        for (int i = 0; i < k; i++) {
            inputReshapeDims[i + k + 1] = pInputDims[i + 1];
            inputReshapeDims[i + 1] = blockSize;
        }
        inputReshapeDims[0] = pInputDims[0] / (blockSize * blockSize);
        inputReshapeDims[2 * k + 1] = pInputDims[k + 1];

        for (int i = 0; i < k; i++) {
            orders[2 * i + 1] = i + 1; // 2 * k + 1 - (k + k - i)
            orders[2 * i + 2] = k + i + 1; // 2 * k + 1 - (k - i)
        }
        orders[0] = 0;
        orders[2 * k + 1] = 2 * k + 1;
    } else if (isNHWCOrder && mode == 1) {
        for (int i = 0; i < k; i++) {
            inputReshapeDims[i + k + 1] = pInputDims[i + 1];
            inputReshapeDims[i] = blockSize;
        }
        inputReshapeDims[k] = pInputDims[0] / (blockSize * blockSize);
        inputReshapeDims[2 * k + 1] = pInputDims[k + 1];

        for (int i = 0; i < k; i++) {
            orders[2 * i] = k + i; // 2 * k + 1 - (k + 1 - i)
            orders[2 * i + 1] = i; // 2 * k + 1 - ((k - i) + k + 1)
        }
        orders[2 * k] = 2 * k;
        orders[2 * k + 1] = 2 * k + 1;
    } else {
        return;
    }

    int32_t outputReshapeDims[MAX_DTS_DIMS];
    int32_t numReshapeDims = 2 * k + 2;
    for (int i = 0; i < numReshapeDims; i++) {
        int32_t idx = orders[i];
        outputReshapeDims[i] = inputReshapeDims[idx];
    }

    int32_t numValues = 1;
    for (int i = 0; i < numInputDims; i++) {
        numValues *= pInputDims[i];
    }
    for (int inIdx = 0; inIdx < numValues; inIdx++) {
        int32_t inputReshapeCoords[MAX_DTS_DIMS] = {0};
        getCoord(inIdx, inputReshapeDims, numReshapeDims, inputReshapeCoords);

        int32_t outputReshapeCoords[MAX_DTS_DIMS] = {0};
        for (int j = 0; j < numReshapeDims; j++) {
            int32_t idx = orders[j];
            outputReshapeCoords[j] = inputReshapeCoords[idx];
        }

        int32_t outIdx = 0;
        int32_t sizeCount = 1;
        for (int j = 0; j < numReshapeDims; j++) {
            outIdx += sizeCount * outputReshapeCoords[j];
            sizeCount *= outputReshapeDims[j];
        }

        half* input = pActInput + inIdx;
        half* output = pActOutput + outIdx;
        *output = *input;
    }
}
}
