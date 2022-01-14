//
// Copyright Intel Corporation.
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

#pragma once

#include "Op.h"

#include <layers/param_custom_cpp.h>
#include <common_types.h>

struct CustomCppLayerParams {
    uint32_t leonPreambleID;

    uint8_t* kernelData;
    size_t kernelDataLen;

    uint32_t* paramData;
    sw_params::BaseKernelParams baseParamData;
    size_t paramDataLen;
    uint64_t kernel = 0;
};

class CustomCpp : public Op
{
public:
    CustomCpp() : Op(kCustomCpp) {};
    CustomCpp(t_MvTensorOpType /*op_type*/) : Op(kCustomCpp) {};
    virtual ~CustomCpp() override;

    virtual void run(mv::tensor::Processor& mvtp,
            t_MvTensorMyriadResources& myriadRes,
            t_MvTensorDebugInfo& debugInfo) override;
    void addInputBuffer(const OpTensor& input, sw_params::Location loc = sw_params::Location::DDR) {
        inputVec.push_back(input);
        inputLocations.push_back(loc);
    }
    void addOutputBuffer(const OpTensor& output, sw_params::Location loc = sw_params::Location::DDR) {
        outputVec.push_back(output);
        outputLocations.push_back(loc);
    }

    virtual bool parse(Layer *layer) override;

    CustomCppLayerParams ops;

private:
    std::vector<OpTensor> inputVec;
    std::vector<sw_params::Location> inputLocations;
    std::vector<OpTensor> outputVec;
    std::vector<sw_params::Location> outputLocations;
};
