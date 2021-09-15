// {% copyright %}

#pragma once

#include "Op.h"

#include <layers/param_custom_cpp.h>

struct CustomCppLayerParams {
    uint32_t leonPreambleID;

    const uint8_t* kernelData;
    size_t kernelDataLen;

    const uint32_t* paramData;
    size_t paramDataLen;
    uint32_t opID = 0;
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
    void addInputBuffer(const Buffer& input) {
        inputVec.push_back(input);
    }
    void addOutputBuffer(const Buffer& output) {
        outputVec.push_back(output);
    }

#ifdef CONFIG_TARGET_SOC_3720
    virtual bool parse(Layer *layer) override;
#endif

    CustomCppLayerParams ops;
    nn::shave_lib::CustomLayerCppParams p;

private:
    std::vector<Buffer> inputVec;
    std::vector<Buffer> outputVec;
};
