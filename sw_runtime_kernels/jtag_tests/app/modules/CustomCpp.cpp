//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <cstring>
#include "CustomCpp.h"
#include <mv_types.h>
#include "shave_task_runner.hpp"

#include <mvMacros.h>
#include <nn_cache.h>
#include <nn_memory.h>
#include <nn_tensor_ref.h>

using namespace nn;
using namespace mv::tensor;

CustomCpp::~CustomCpp(){}

void CustomCpp::run() {
    ShaveTaskRunner runner;

    sw_params::MemRefData* inTensors =
            reinterpret_cast<sw_params::MemRefData*>(reinterpret_cast<uint8_t*>(ops.paramData) + ops.baseParamData.inputsOffset);
    sw_params::MemRefData* outTensors =
            reinterpret_cast<sw_params::MemRefData*>(reinterpret_cast<uint8_t*>(ops.paramData) + ops.baseParamData.outputsOffset);

    for (unsigned i = 0; i < inputVec.size(); i++) {
        inTensors[i] = inputVec[i].toMemRefData(inputLocations[i], true);
        nn::cache::flush(inTensors[i]);
    }

    for (unsigned i = 0; i < outputVec.size(); i++) {
        outTensors[i] = outputVec[i].toMemRefData( outputLocations[i], false);
        nn::cache::flush(outTensors[i]);
    }

    mvTensorAssert(runner.enqueTask(ops), "custom OpenCPP layer run failed");
    mvTensorAssert(runner.dequeResult(), "custom Cpp layer run failed");

    for (unsigned i = 0; i < outputVec.size(); i++) {
        if (outTensors[i].location == sw_params::Location::NN_CMX) {
            auto totalBytes = (outputVec[i].ndims > 0) ? outputVec[i].dims[outputVec[i].ndims - 1] * outputVec[i].strides[outputVec[i].ndims - 1] : 0;
            nn::cache::invalidate(reinterpret_cast<uint8_t*>(outTensors[i].dataAddr), totalBytes);
            memcpy(reinterpret_cast<uint8_t*>(outputVec[i].addr), reinterpret_cast<uint8_t*>(outTensors[i].dataAddr), totalBytes);
            nn::cache::flush(reinterpret_cast<uint8_t*>(outputVec[i].addr), totalBytes);
        }
    }

    // Restart cmx allocation after test is done
    resetReservedAllocation();
}
