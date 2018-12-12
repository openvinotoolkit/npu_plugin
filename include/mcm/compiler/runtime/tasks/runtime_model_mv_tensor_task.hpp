#ifndef MV_RUNTIME_MODEL_MV_TENSOR_TASK_
#define MV_RUNTIME_MODEL_MV_TENSOR_TASK_

#include "include/mcm/compiler/runtime/tasks/runtime_model_specific_task.hpp"
#include "include/mcm/compiler/runtime/runtime_model_tensor_reference.hpp"

#include <vector>

namespace mv
{
    struct RuntimeModelCustomSoftware : public RuntimeModelSoftwareLayer
    {
        std::vector<unsigned> data;
        unsigned lenght;
        unsigned id;
    };

    struct RuntimeModelPassthroughSoftware : public RuntimeModelSoftwareLayer
    {
        RuntimeModelTensorReference * input;
        RuntimeModelTensorReference * output;
    };

    struct RuntimeModelReLuSoftware : public RuntimeModelSoftwareLayer
    {
        unsigned opX;
        RuntimeModelTensorReference * input;
        RuntimeModelTensorReference * output;
        unsigned strideX;
        unsigned strideY;
    };

    struct RuntimeModelPoolingSoftware : public RuntimeModelSoftwareLayer
    {
        unsigned radixX;
        unsigned radixY;
        unsigned strideX;
        unsigned strideY;
        unsigned padX;
        unsigned padY;
        unsigned padStyle;
        unsigned dilation;
        RuntimeModelTensorReference * input;
        RuntimeModelTensorReference * output;
    };

    struct RuntimeModelConv2DSoftware : public RuntimeModelSoftwareLayer
    {
        unsigned radixX;
        unsigned radixY;
        unsigned strideX;
        unsigned strideY;
        unsigned padX;
        unsigned padY;
        unsigned padStyle;
        unsigned dilation;
        RuntimeModelTensorReference * input;
        RuntimeModelTensorReference * output;
        RuntimeModelTensorReference * weight;
        RuntimeModelTensorReference * bias;
    };

    struct RuntimeModelSoftwareLayer
    {

    };

    struct RuntimeModelMvTensorTask : public RuntimeModelSpecificTask
    {
        RuntimeModelSoftwareLayer * layer;
    };
}

#endif //MV_RUNTIME_MODEL_MV_TENSOR_TASK_
