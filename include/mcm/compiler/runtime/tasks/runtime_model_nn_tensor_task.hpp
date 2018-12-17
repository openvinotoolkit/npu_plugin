#ifndef MV_RUNTIME_MODEL_NN_TENSOR_TASK_
#define MV_RUNTIME_MODEL_NN_TENSOR_TASK_

#include "include/mcm/compiler/runtime/tasks/runtime_model_task.hpp"
#include "include/mcm/compiler/runtime/runtime_model_tensor_reference.hpp"
#include "KeemBayFBSchema/compiledSchemas/nnController_generated.h"

namespace mv
{
    struct RuntimeModelPPEAssist : public RuntimeModelPPEHelper
    {
        unsigned op;
    };

    flatbuffers::Offset<PPEConfigure> convertToFlatbuffer(RuntimeModelPPEAssist * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return MVCNN::CreatePPEAssist(fbb, ref->op);
    }

    struct RuntimeModelPPEConfigure : public RuntimeModelPPEHelper
    {
        std::vector<unsigned> * vals;
    };

    flatbuffers::Offset<PPEAssist> convertToFlatbuffer(RuntimeModelPPEAssist * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return MVCNN::CreatePPEAssistDirect(fbb, ref->vals);
    }

    struct RuntimeModelPPEHelper
    {

    };

    flatbuffers::Offset<void> convertToFlatbuffer(RuntimeModelPPEHelper * ref, RuntimeModelPPEHelperType type, flatbuffers::FlatBufferBuilder& fbb)
    {
        switch (type)
        {
            case NONE:
                return convertToFlatbuffer((RuntimeModelPPEAssist *) ref, flatbuffers::FlatBufferBuilder& fbb);
                break;
            case PPEASSIST:
                return convertToFlatbuffer((RuntimeModelPPEAssist *) ref, flatbuffers::FlatBufferBuilder& fbb);
                break;
            case PPECONFIGURE:
                return convertToFlatbuffer((RuntimeModelPPEConfigure *) ref, flatbuffers::FlatBufferBuilder& fbb);
                break;
            default:
                return convertToFlatbuffer((RuntimeModelPPEConfigure *) ref, flatbuffers::FlatBufferBuilder& fbb);
                break;
        }
    }

    enum RuntimeModelPPEHelperType
    {
        NONE,
        PPEASSIST,
        PPECONFIGURE
    };

    struct RuntimeModelNNTask : public RuntimeModelSpecificTask
    {
        RuntimeModelPPEHelper * subtask;
        RuntimeModelPPEHelperType subtaskType;
    };

    flatbuffers::Offset<void> convertToFlatbuffer(RuntimeModelNNTask * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return CreateNNTensorTask(fbb, ref->subtaskType, convertToFlatbuffer(ref->subtask, ref->subtaskType, fbb));
    }

}

#endif //MV_RUNTIME_MODEL_NN_TENSOR_TASK_
