#ifndef MV_RUNTIME_MODEL_TASK_
#define MV_RUNTIME_MODEL_TASK_

#include <vector>
#include "include/mcm/compiler/runtime/runtime_model_barrier.hpp"
#include "KeemBayFBSchema/compiledSchemas/graphfile_generated.h"

//Forward declaration of subclasses to avoid circular inclusion problem

struct RuntimeModelControllerTask;
struct RuntimeModelUPADMATask;
struct RuntimeModelNNDMATask;
struct RuntimeModelDPUTask;
struct RuntimeModelNCE1Task;
struct RuntimeModelNNTask;

flatbuffers::Offset<void> convertToFlatbuffer(RuntimeModelControllerTask * ref, flatbuffers::FlatBufferBuilder& fbb);
flatbuffers::Offset<void> convertToFlatbuffer(RuntimeModelUPADMATask * ref, flatbuffers::FlatBufferBuilder& fbb);
flatbuffers::Offset<void> convertToFlatbuffer(RuntimeModelNNDMATask * ref, flatbuffers::FlatBufferBuilder& fbb);
flatbuffers::Offset<void> convertToFlatbuffer(RuntimeModelDPUTask * ref, flatbuffers::FlatBufferBuilder& fbb);
flatbuffers::Offset<void> convertToFlatbuffer(RuntimeModelNCE1Task * ref, flatbuffers::FlatBufferBuilder& fbb);
flatbuffers::Offset<void> convertToFlatbuffer(RuntimeModelNNTask * ref, flatbuffers::FlatBufferBuilder& fbb);

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

    flatbuffers::Offset<void> convertToFlatbuffer(RuntimeModelSpecificTask * ref, RuntimeModelSpecificTaskType taskType, flatbuffers::FlatBufferBuilder& fbb)
    {
        switch (taskType)
        {
            case NONETASK:
                return convertToFlatbuffer((RuntimeModelControllerTask *) ref, fbb);
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

    flatbuffers::Offset<MVCNN::Task> convertToFlatbuffer(RuntimeModelTask * ref, flatbuffers::FlatBufferBuilder&fbb)
    {
        return MVCNN::CreateTaskDirect(
            fbb,
            ref->nodeID,
            ref->sourceTaskIDs,
            convertToFlatbuffer(ref->associatedBarriers, fbb),
            static_cast<MVCNN::SpecificTask>(ref->taskType),
            convertToFlatbuffer(ref->task, ref->taskType, fbb));
    }

    flatbuffers::Offset<MVCNN::TaskList> convertToFlatbuffer(std::vector<RuntimeModelTask*> * ref, flatbuffers::FlatBufferBuilder&fbb)
    {
        std::vector<flatbuffers::Offset<MVCNN::Task>> * content = new std::vector<flatbuffers::Offset<MVCNN::Task>>();
        for(unsigned i = 0; i < ref->size(); ++i)
        {
            RuntimeModelTask* currentRef = ref->at(i);
            flatbuffers::Offset<MVCNN::Task> currentOffset = convertToFlatbuffer(currentRef, fbb);
            content->push_back(currentOffset);
        }
        return MVCNN::CreateTaskListDirect(fbb, content);
    }

    std::vector<flatbuffers::Offset<MVCNN::TaskList>> * convertToFlatbuffer(std::vector<std::vector<RuntimeModelTask*>*> * ref, flatbuffers::FlatBufferBuilder&fbb)
    {
        std::vector<flatbuffers::Offset<MVCNN::TaskList>> * taskLists = new std::vector<flatbuffers::Offset<MVCNN::TaskList>>();
        for(unsigned i = 0; i < ref->size(); ++i)
        {
            std::vector<RuntimeModelTask*> * currentRef = ref->at(i);
            flatbuffers::Offset<MVCNN::TaskList> currentOffset = convertToFlatbuffer(currentRef, fbb);
            taskLists->push_back(currentOffset);
        }
        return taskLists;
    }
}

#endif
