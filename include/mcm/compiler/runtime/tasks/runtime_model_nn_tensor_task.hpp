#ifndef MV_RUNTIME_MODEL_NN_TENSOR_TASK_
#define MV_RUNTIME_MODEL_NN_TENSOR_TASK_

#include "include/mcm/compiler/runtime/tasks/runtime_model_specific_task.hpp"
#include "include/mcm/compiler/runtime/runtime_model_tensor_reference.hpp"

namespace mv
{
    struct RuntimeModelPPEAssist : public RuntimeModelPPEHelper
    {
        unsigned op;
    };

    struct RuntimeModelPPEConfigure : public RuntimeModelPPEHelper
    {
        std::vector<unsigned> vals;
    };

    struct RuntimeModelPPEHelper
    {

    };

    struct RuntimeModelNNTask : public RuntimeModelSpecificTask
    {
        RuntimeModelPPEHelper * subtask;
    };
}

#endif //MV_RUNTIME_MODEL_NN_TENSOR_TASK_
