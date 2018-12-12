#ifndef MV_RUNTIME_MODEL_CONTROLLER_TASK_
#define MV_RUNTIME_MODEL_CONTROLLER_TASK_

#include "include/mcm/compiler/runtime/tasks/runtime_model_specific_task.hpp"
#include "include/mcm/compiler/runtime/runtime_model_tensor_reference.hpp"
#include "include/mcm/compiler/runtime/runtime_model_barrier.hpp"

namespace mv
{
    struct RuntimeModelBarrierConfigurationTask : public RuntimeModelControllerSubTask
    {
        RuntimeModelBarrier * target;
    };

    struct RuntimeModelMemoryTask : public RuntimeModelControllerSubTask
    {
        unsigned id;
    };

    struct RuntimeModelTimerTask : public RuntimeModelControllerSubTask
    {
        unsigned id;
        RuntimeModelTensorReference * writeLocation;
    };

    struct RuntimeModelControllerSubTask
    {

    };

    struct RuntimeModelControllerTask : public RuntimeModelSpecificTask
    {
        RuntimeModelControllerSubTask * task;
    };
}

#endif
