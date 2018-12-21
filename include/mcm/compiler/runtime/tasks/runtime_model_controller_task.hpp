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

        flatbuffers::Offset<void> convertToFlatbuffer(flatbuffers::FlatBufferBuilder& fbb)
        {
            return MVCNN::CreateBarrierConfigurationTask(fbb, convertToFlatbuffer(fbb, ref->target)).Union();
        }
    };

    struct RuntimeModelMemoryTask : public RuntimeModelControllerSubTask
    {
        unsigned id;

        flatbuffers::Offset<void> convertToFlatbuffer(flatbuffers::FlatBufferBuilder& fbb)
        {
            return MVCNN::CreateMemoryTask(fbb, id).Union();
        }
    };

    struct RuntimeModelTimerTask : public RuntimeModelControllerSubTask
    {
        unsigned id;
        RuntimeModelTensorReference * writeLocation;

        flatbuffers::Offset<void> convertToFlatbuffer(flatbuffers::FlatBufferBuilder& fbb)
        {
            return MVCNN::CreateTimerTask(fbb, ref->id, convertToFlatbuffer(ref->writeLocation, fbb)).Union();
        }
    };

    struct RuntimeModelControllerSubTask : public Flatbufferizable
    {
        virtual flatbuffers::Offset<void> convertToFlatbuffer(flatbuffers::FlatBufferBuilder& fbb) = 0;
    };

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

        flatbuffers::Offset<void> convertToFlatbuffer(flatbuffers::FlatBufferBuilder& fbb)
        {
            return CreateControllerTask(
                        fbb,
                        taskType,
                        task->convertToFlatbuffer(fbb)).Union();
        }
    };



}

#endif
