#ifndef MV_RUNTIME_MODEL_CONTROLLER_TASK_
#define MV_RUNTIME_MODEL_CONTROLLER_TASK_

#include "include/mcm/compiler/runtime/tasks/runtime_model_task.hpp"
#include "include/mcm/compiler/runtime/runtime_model_tensor_reference.hpp"
#include "include/mcm/compiler/runtime/runtime_model_barrier.hpp"
#include "KeemBayFBSchema/compiledSchemas/nnController_generated.h"

namespace mv
{
    struct RuntimeModelBarrierConfigurationTask : public RuntimeModelControllerSubTask
    {
        RuntimeModelBarrier * target;
    };

    flatbuffers::Offset<MVCNN::BarrierConfigurationTask> convertToFlatbuffer(RuntimeModelBarrierConfigurationTask * ref, flatbuffers::FlatBufferBuilder * fbb)
    {
        return MVCNN::CreateBarrierConfigurationTask(fbb, convertToFlatbuffer(fbb, ref->target));
    }

    struct RuntimeModelMemoryTask : public RuntimeModelControllerSubTask
    {
        unsigned id;
    };

    flatbuffers::Offset<MVCNN::MemoryTask> convertToFlatbuffer(RuntimeModelMemoryTask * ref, flatbuffers::FlatBufferBuilder * fbb)
    {
        return MVCNN::CreateMemoryTask(fbb, ref->id);
    }

    struct RuntimeModelTimerTask : public RuntimeModelControllerSubTask
    {
        unsigned id;
        RuntimeModelTensorReference * writeLocation;
    };

    flatbuffers::Offset<MVCNN::TimerTask> convertToFlatbuffer(RuntimeModelTimerTask * ref, flatbuffers::FlatBufferBuilder * fbb)
    {
        return MVCNN::CreateTimerTask(fbb, ref->id, convertToFlatbuffer(ref->writeLocation, fbb));
    }

    struct RuntimeModelControllerSubTask
    {

    };

    flatbuffers::Offset<void> convertToFlatbuffer(RuntimeModelControllerSubTask * ref, RuntimeModelControllerSubTaskType taskType, flatbuffers::FlatBufferBuilder * fbb)
    {
        switch (taskType)
        {
            case NONETYPE:
                return convertToFlatbuffer((RuntimeModelTimerTask *) ref, fbb);
                break;
            case BARRIERCONFIGURATION:
                return convertToFlatbuffer((RuntimeModelBarrierConfigurationTask *) ref, fbb);
                break;
            case MEMORY:
                return convertToFlatbuffer((RuntimeModelMemoryTask *) ref, fbb);
                break;
            case TIMER:
                return convertToFlatbuffer((RuntimeModelTimerTask *) ref, fbb);
                break;
            default:
                break;
        }
    }

    enum RuntimeModelControllerSubTaskType
    {
        NONETYPE,
        BARRIERCONFIGURATION,
        MEMORY,
        TIMER
    };

    struct RuntimeModelControllerTask : public RuntimeModelSpecificTask
    {
        RuntimeModelControllerSubTaskType taskType;
        RuntimeModelControllerSubTask * task;
    };

    flatbuffers::Offset<void> convertToFlatbuffer(RuntimeModelControllerTask * ref, flatbuffers::FlatBufferBuilder * fbb)
    {
        return CreateControllerTask(
                    fbb,
                    RuntimeModelControllerSubTaskType,
                    convertToFlatbuffer(ref->task, ref->taskType, fbb));
    }

}

#endif
