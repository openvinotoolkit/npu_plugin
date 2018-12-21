#ifndef MV_RUNTIME_MODEL_TASK_
#define MV_RUNTIME_MODEL_TASK_

#include <vector>
#include "include/mcm/compiler/runtime/runtime_model_barrier.hpp"
#include "meta/schema/graphfile/graphfile_generated.h"
#include "include/mcm/compiler/runtime/flatbufferizable.hpp"

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

    struct RuntimeModelSpecificTask : public Flatbufferizable
    {
        virtual flatbuffers::Offset<void> convertToFlatbuffer(flatbuffers::FlatBufferBuilder &fbb) = 0;
    };

    struct RuntimeModelTask
    {
        unsigned nodeID;
        std::vector<unsigned> * sourceTaskIDs;
        RuntimeModelBarrierReference * associatedBarriers;
        RuntimeModelSpecificTaskType taskType;
        RuntimeModelSpecificTask * task;
    };


    //Task to flatbuffer
    flatbuffers::Offset<MVCNN::Task> convertToFlatbuffer(RuntimeModelTask * ref, flatbuffers::FlatBufferBuilder&fbb)
    {
        return MVCNN::CreateTaskDirect(
            fbb,
            ref->nodeID,
            ref->sourceTaskIDs,
            convertToFlatbuffer(ref->associatedBarriers, fbb),
            static_cast<MVCNN::SpecificTask>(ref->taskType),
            ref->task->convertToFlatbuffer(fbb));
    }

    //Vector of Tasks to TaskLists
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

    //Vector of TasksLists to flatbuffer
    std::vector<flatbuffers::Offset<MVCNN::TaskList>> * convertToFlatbuffer(std::vector<std::vector<RuntimeModelTask*>*> * ref, flatbuffers::FlatBufferBuilder&fbb)
    {
        std::vector<flatbuffers::Offset<MVCNN::TaskList>> * taskLists = new std::vector<flatbuffers::Offset<MVCNN::TaskList>>();
        if(ref)
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
