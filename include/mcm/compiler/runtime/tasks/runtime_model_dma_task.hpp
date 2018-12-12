#ifndef MV_RUNTIME_MODEL_DMA_TASK_
#define MV_RUNTIME_MODEL_DMA_TASK_

#include "include/mcm/compiler/runtime/tasks/runtime_model_specific_task.hpp"
#include "include/mcm/compiler/runtime/runtime_model_tensor_reference.hpp"

#include <vector>

namespace mv
{

    struct RuntimeModelUPADMATask : public RuntimeModelSpecificTask
    {
        RuntimeModelTensorReference * src;
        RuntimeModelTensorReference * dst;
    };

    struct RuntimeModelNNDMATask : public RuntimeModelSpecificTask
    {
        std::vector<RuntimeModelTensorReference*> dst;
        RuntimeModelTensorReference * src;
        bool compression;
    };
}

#endif
