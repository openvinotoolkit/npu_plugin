#ifndef MV_RUNTIME_MODEL_TASK_
#define MV_RUNTIME_MODEL_TASK_

#include <vector>
#include "include/mcm/compiler/runtime/runtime_model_barrier.hpp"

namespace mv
{
    enum RuntimeModelSpecificTaskType
    {
        NONETASK,
        CONTROLLERTASK,
        UPADMATASK,
        NNDMATASK,
        DPUTASK,
        NCE1TASK,
        NNTENSORTASK
    };

    struct RuntimeModelSpecificTask
    {

    };

    flatbuffers::Offset<void> convertToFlatbuffer(RuntimeModelSpecificTask * ref, RuntimeModelSpecificTaskType taskType, flatbuffers::FlatBufferBuilder *fbb)
    {
        switch (taskType)
        {
            case NONETASK:
                break;
            case CONTROLLERTASK:
                return convertToFlatbuffer((RuntimeModelControllerTask *) ref, fbb);
                break;
            case UPADMATASK:
                return convertToFlatbuffer((RuntimeModelUPADMATask *) ref, fbb);
                break;
            case NNDMATASK:
                return convertToFlatbuffer((RuntimeModelNNDMATask *) ref, fbb);
                break;
            case DPUTASK:
                return convertToFlatbuffer((RuntimeModelDPUTask *) ref, fbb);
                break;
            case NCE1TASK:
                return convertToFlatbuffer((RuntimeModelNCE1Task *) ref, fbb);
                break;
            case NNTENSORTASK:
                return convertToFlatbuffer((RuntimeModelNNTask *) ref, fbb);
                break;
            default:
                break;
        }
    }

    struct RuntimeModelTask
    {
        unsigned nodeID;
        std::vector<unsigned> * sourceTaskIDs;
        RuntimeModelBarrierReference * associatedBarriers;
        RuntimeModelSpecificTaskType taskType;
        RuntimeModelSpecificTask * task;
    };

    flatbuffers::Offset<Task> convertToFlatbuffer(RuntimeModelTask * ref, flatbuffers::FlatBufferBuilder *fbb)
    {
        return CreateTaskDirect(
            fbb,
            ref->nodeID,
            ref->sourceTaskIDs,
            convertToFlatBuffer(ref->associatedBarriers, fbb),
            ref->taskType,
            convertToFlatbuffer(ref->task, ref->taskType, fbb));
    }
}

#endif
