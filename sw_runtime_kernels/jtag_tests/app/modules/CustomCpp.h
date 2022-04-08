//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "Op.h"
#include <common_types.h>
#include <nn_log.h>
#include <elf.h>

constexpr uint8_t MAX_INPUT_TENSORS = 8;
constexpr uint8_t MAX_OUTPUT_TENSORS = 8;

struct __attribute__((aligned(64))) CustomCppLayerParams {
    uint32_t* paramData;
    sw_params::BaseKernelParams baseParamData;
    uint32_t kernel = 0;
    uint32_t leonPreambleID;
    size_t kernelDataLen;
    size_t paramDataLen;
};

class CustomCpp : public Op
{
public:
    CustomCpp() : Op(kCustomCpp) {};
    CustomCpp(t_MvTensorOpType /*op_type*/) : Op(kCustomCpp) {};
    virtual ~CustomCpp();

    virtual void run() override;
    
    void addInputBuffer(const OpTensor& input, sw_params::Location loc = sw_params::Location::DDR) {
        inputVec.push_back(input);
        inputLocations.push_back(loc);
    }
    void addOutputBuffer(const OpTensor& output, sw_params::Location loc = sw_params::Location::DDR) {
        outputVec.push_back(output);
        outputLocations.push_back(loc);
    }

    CustomCppLayerParams ops;

private:
    std::vector<OpTensor> inputVec;
    std::vector<sw_params::Location> inputLocations;
    std::vector<OpTensor> outputVec;
    std::vector<sw_params::Location> outputLocations;
};
