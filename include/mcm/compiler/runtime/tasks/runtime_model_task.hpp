#ifndef MV_RUNTIME_MODEL_TASK_
#define MV_RUNTIME_MODEL_TASK_

#include <vector>
#include "include/mcm/compiler/runtime/runtime_model_barrier.hpp"
#include "include/mcm/compiler/runtime/tasks/runtime_model_specific_task.hpp"

namespace mv
{
    struct RuntimeModelTask
    {
        unsigned nodeID;
        std::vector<unsigned> * sourceTaskIDs;
        RuntimeModelBarrierReference * associatedBarriers;
        RuntimeModelSpecificTask * task;
    };
}

#endif
