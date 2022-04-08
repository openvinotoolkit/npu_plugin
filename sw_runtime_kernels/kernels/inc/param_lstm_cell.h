//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#ifdef __MOVICOMPILE__
#include <moviVectorTypes.h>
#else
typedef fp16 half;
#endif

#include <common_types.h>

#ifdef __cplusplus
namespace sw_params {
#endif

struct LSTMCellParams {
    // Inputs
    struct MemRefData inputData;
    struct MemRefData initialHiddenState;
    struct MemRefData initialCellState;
    struct MemRefData weights;
    struct MemRefData recurrenceWeights;
    struct MemRefData biases;

    // Outputs
    struct MemRefData outputHiddenState;
    struct MemRefData outputCellState;

    int64_t hiddenSize;
    int64_t RNNForward;
    int64_t nCells;
    int64_t nBatches;
    int64_t outputsNumber;
    int64_t useCellState;
};

inline struct BaseKernelParams ToBaseKernelParams(struct LSTMCellParams* params) {
    struct BaseKernelParams result;
    result.numInputs = 6;
    result.numOutputs = 2;
#ifdef __cplusplus
    result.inputsOffset = reinterpret_cast<uint8_t*>(&(params->inputData)) - reinterpret_cast<uint8_t*>(params);
    result.outputsOffset =
            reinterpret_cast<uint8_t*>(&(params->outputHiddenState)) - reinterpret_cast<uint8_t*>(params);
#else
    result.inputsOffset = (uint8_t*)(&(params->inputData)) - (uint8_t*)(params);
    result.outputsOffset = (uint8_t*)(&(params->outputHiddenState)) - (uint8_t*)(params);
#endif
    return result;
}

#ifdef __cplusplus
}  // namespace sw_params
#endif
