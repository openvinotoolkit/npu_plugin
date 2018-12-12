#ifndef MV_RUNTIME_MODEL_NCE1_TASK_
#define MV_RUNTIME_MODEL_NCE1_TASK_

#include "include/mcm/compiler/runtime/tasks/runtime_model_specific_task.hpp"
#include "include/mcm/compiler/runtime/runtime_model_tensor_reference.hpp"

namespace mv
{
    struct RuntimeModelNCE1Tensor
    {
        unsigned dimX;
        unsigned dimY;
        unsigned dimZ;
        unsigned strideX;
        unsigned strideY;
        unsigned offset;
        unsigned location;
        unsigned datatype;
        unsigned order;
    };

    struct RuntimeModelNCE1FullyConnected : public RuntimeModelNCE1Layer
    {
        unsigned streamingMask;
        unsigned inputSize;
        unsigned outputSize;
        unsigned concatOffset;
        unsigned unloadCMX;
        unsigned overwriteInput;
        unsigned CMXSize_;
        unsigned reluSHVAcc_;
        unsigned shvNegSlope;
        unsigned shvPosSlope;
        unsigned desc_count;
        std::vector<unsigned> descriptors;
        RuntimeModelNCE1Tensor * input;
        RuntimeModelNCE1Tensor * output;
        RuntimeModelNCE1Tensor * weight;
        RuntimeModelNCE1Tensor * bias;
    };

    struct RuntimeModelNCE1Pool : public RuntimeModelNCE1Layer
    {
        unsigned streamingMask;
        unsigned inputSize;
        unsigned outputSize;
        unsigned concatOffset;
        unsigned unloadCMX;
        unsigned overwriteInput;
        unsigned CMXSize;
        unsigned reluSHVAcc;
        unsigned shvNegSlope;
        unsigned shvPosSlope;
        unsigned desc_count;
        std::vector<unsigned> descriptors;
        RuntimeModelNCE1Tensor * input;
        RuntimeModelNCE1Tensor * output;
        RuntimeModelNCE1Tensor * weight;
        RuntimeModelNCE1Tensor * bias;
    };

    struct RuntimeModelNCE1Conv : public RuntimeModelNCE1Layer
    {
        unsigned streamingMask;
        unsigned inputSize;
        unsigned outputSize;
        unsigned concatOffset;
        unsigned unloadCMX;
        unsigned overwriteInput;
        unsigned CMXSize;
        unsigned reluSHVAcc;
        unsigned shvNegSlope;
        unsigned shvPosSlope;
        unsigned descCount;
        std::vector<unsigned> descriptors;
        RuntimeModelNCE1Tensor * input;
        RuntimeModelNCE1Tensor * output;
        RuntimeModelNCE1Tensor * weight;
        RuntimeModelNCE1Tensor * bias;
    };

    struct RuntimeModelNCE1Layer
    {

    };

    struct RuntimeModelNCE1Task : public RuntimeModelSpecificTask
    {
        RuntimeModelNCE1Layer * layer;
    };
}

#endif
