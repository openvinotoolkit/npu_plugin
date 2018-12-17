#ifndef MV_RUNTIME_MODEL_DMA_TASK_
#define MV_RUNTIME_MODEL_DMA_TASK_

#include "include/mcm/compiler/runtime/tasks/runtime_model_task.hpp"
#include "include/mcm/compiler/runtime/runtime_model_tensor_reference.hpp"
#include "KeemBayFBSchema/compiledSchemas/dma_generated.h"

#include <vector>

namespace mv
{

    struct RuntimeModelUPADMATask : public RuntimeModelSpecificTask
    {
        RuntimeModelTensorReference * src;
        RuntimeModelTensorReference * dst;
    };

    flatbuffers::Offset<MVCNN::UPADMATask> convertToFlatbuffer(RuntimeModelUPADMATask * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return MVCNN::CreateUPADMATask(fbb, convertToFlatbuffer(ref->src, fbb), convertToFlatbuffer(ref->dst, fbb));
    }

    struct RuntimeModelNNDMATask : public RuntimeModelSpecificTask
    {
        std::vector<RuntimeModelTensorReference*> dst;
        RuntimeModelTensorReference * src;
        bool compression;
    };

    flatbuffers::Offset<MVCNN::NNDMATask> convertToFlatbuffer(RuntimeModelNNDMATask * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return MVCNN::CreateNNDMATaskDirect(fbb, convertToFlatbuffer(ref->src, fbb), convertToFlatbuffer(ref->dst, fbb), ref->compression);
    }
}

#endif
