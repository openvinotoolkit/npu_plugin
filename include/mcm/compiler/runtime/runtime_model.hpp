#ifndef MV_RUNTIME_MODEL_
#define MV_RUNTIME_MODEL_

#include "include/mcm/compiler/runtime/runtime_model_header.hpp"
#include "include/mcm/compiler/runtime/tasks/runtime_model_task.hpp"
#include "include/mcm/compiler/runtime/runtime_model_barrier.hpp"
#include "include/mcm/compiler/runtime/runtime_model_binary_data.hpp"
#include "KeemBayFBSchema/compiledSchemas/graphfile_generated.h"
#include <vector>

namespace mv
{
    struct RuntimeModel
    {
        RuntimeModelHeader * header;
        std::vector<std::vector<RuntimeModelTask*>*> * taskLists;
        std::vector<RuntimeModelBarrier*> * barrierTable;
        std::vector<RuntimeModelBinaryData*> * binaryData;
    };

    flatbuffers::Offset<MVCNN::GraphFile> convertToFlatbuffer(RuntimeModel * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return MVCNN::CreateGraphFileDirect(fbb,
                                     convertToFlatbuffer(ref->header, fbb),
                                     convertToFlatbuffer(ref->taskLists, fbb),
                                     convertToFlatbuffer(ref->barrierTable, fbb),
                                     convertToFlatbuffer(ref->binaryData, fbb));
    }
}

#endif
