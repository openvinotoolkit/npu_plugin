#ifndef MV_RUNTIME_MODEL_MV_TENSOR_TASK_
#define MV_RUNTIME_MODEL_MV_TENSOR_TASK_

#include "include/mcm/compiler/runtime/tasks/runtime_model_task.hpp"
#include "include/mcm/compiler/runtime/runtime_model_tensor_reference.hpp"
#include "KeemBayFBSchema/compiledSchemas/software_generated.h"

#include <vector>

namespace mv
{
    struct RuntimeModelCustomSoftware : public RuntimeModelSoftwareLayer
    {
        std::vector<unsigned> * data;
        std::vector<unsigned> * lenght;
        unsigned id;
    };

    flatbuffers::Offset<MVCNN::Custom> convertToFlatbuffer(RuntimeModelCustomSoftware * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return MVCNN::CreateCustomDirect(fbb,
                                             ref->data,
                                             ref->lenght,
                                             id);
    }

    struct RuntimeModelPassthroughSoftware : public RuntimeModelSoftwareLayer
    {
        RuntimeModelTensorReference * input;
        RuntimeModelTensorReference * output;
    };

    flatbuffers::Offset<MVCNN::Passthrough> convertToFlatbuffer(RuntimeModelPassthroughSoftware * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return MVCNN::CreatePassThrough(
            fbb,
            convertToFlatbuffer(ref->input, fbb),
            convertToFlatbuffer(ref->output, fbb));
    }

    struct RuntimeModelReLuSoftware : public RuntimeModelSoftwareLayer
    {
        unsigned opX;
        RuntimeModelTensorReference * input;
        RuntimeModelTensorReference * output;
        unsigned strideX;
        unsigned strideY;
    };

    flatbuffers::Offset<MVCNN::ReLU> convertToFlatbuffer(RuntimeModelReLuSoftware * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return MVCNN::CreateReLU(
            fbb,
            ref->opX,
            convertToFlatbuffer(ref->input, fbb),
            convertToFlatbuffer(ref->output, fbb),
            ref->strideX,
            ref->strideY);
    }

    struct RuntimeModelPoolingSoftware : public RuntimeModelSoftwareLayer
    {
        unsigned radixX;
        unsigned radixY;
        unsigned strideX;
        unsigned strideY;
        unsigned padX;
        unsigned padY;
        unsigned padStyle;
        unsigned dilation;
        RuntimeModelTensorReference * input;
        RuntimeModelTensorReference * output;
    };

    flatbuffers::Offset<MVCNN::Pooling> convertToFlatbuffer(RuntimeModelPoolingSoftware * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return MVCNN::CreatePooling(
            fbb,
            ref->radixX,
            ref->radixY,
            ref->strideX,
            ref->strideY,
            ref->padX,
            ref->padY,
            ref->padStyle,
            ref->dilation,
            convertToFlatbuffer(ref->input, fbb),
            convertToFlatbuffer(ref->output, fbb));
    }

    struct RuntimeModelConv2DSoftware : public RuntimeModelSoftwareLayer
    {
        unsigned radixX;
        unsigned radixY;
        unsigned strideX;
        unsigned strideY;
        unsigned padX;
        unsigned padY;
        unsigned padStyle;
        unsigned dilation;
        RuntimeModelTensorReference * input;
        RuntimeModelTensorReference * output;
        RuntimeModelTensorReference * weight;
        RuntimeModelTensorReference * bias;
    };

    flatbuffers::Offset<MVCNN::Conv2D> convertToFlatbuffer(RuntimeModelConv2DSoftware * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return MVCNN::CreateConv2D(
            fbb,
            ref->radixX,
            ref->radixY,
            ref->strideX,
            ref->strideY,
            ref->padX,
            ref->padY,
            ref->padStyle,
            ref->dilation,
            convertToFlatbuffer(ref->input, fbb),
            convertToFlatbuffer(ref->output, fbb),
            convertToFlatbuffer(ref->weight, fbb),
            convertToFlatbuffer(ref->bias, fbb));
    }

    struct RuntimeModelSoftwareLayer
    {

    };

    flatbuffers::Offset<void> convertToFlatbuffer(RuntimeModelSoftwareLayer * ref, RuntimeModelSoftwareLayerTaskType taskType, flatbuffers::FlatBufferBuilder& fbb)
    {
        switch (ref-taskType)
        {
            case NONE:
                return convertToFlatbuffer((RuntimeModelConv2DSoftware *) ref, flatbuffers::FlatBufferBuilder& fbb);
                break;
            case CONV2D:
                return convertToFlatbuffer((RuntimeModelPoolingSoftware *) ref, flatbuffers::FlatBufferBuilder& fbb);
                break;
            case POOL:
                return convertToFlatbuffer((RuntimeModelPoolingSoftware *) ref, flatbuffers::FlatBufferBuilder& fbb);
                break;
            case RELU:
                return convertToFlatbuffer((RuntimeModelReLuSoftware *) ref, flatbuffers::FlatBufferBuilder& fbb);
                break;
            case PASSTHROUGH:
                return convertToFlatbuffer((RuntimeModelPassthroughSoftware *) ref, flatbuffers::FlatBufferBuilder& fbb);
                break;
            case CUSTOM:
                return convertToFlatbuffer((RuntimeModelCustomSoftware *) ref, flatbuffers::FlatBufferBuilder& fbb);
                break;
            default:
                break;
        }
    }


    enum RuntimeModelSoftwareLayerTaskType
    {
        NONE,
        CONV2D,
        POOL,
        RELU,
        PASSTHROUGH,
        CUSTOM,
    };

    struct RuntimeModelMvTensorTask : public RuntimeModelSpecificTask
    {
        RuntimeModelSoftwareLayer * layer;
        RuntimeModelSoftwareLayerTaskType taskType;
    };

    flatbuffers::Offset<void> convertToFlatbuffer(RuntimeModelMvTensorTask * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return CreateMvTensorTask(
                    fbb,
                    ref->layer,
                    convertToFlatbuffer(ref->layer, ref->taskType, fbb));
    }

}

#endif //MV_RUNTIME_MODEL_MV_TENSOR_TASK_
