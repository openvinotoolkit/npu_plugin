#ifndef MV_RUNTIME_MODEL_NCE1_TASK_
#define MV_RUNTIME_MODEL_NCE1_TASK_

#include "include/mcm/compiler/runtime/tasks/runtime_model_task.hpp"
#include "include/mcm/compiler/runtime/runtime_model_tensor_reference.hpp"
#include "meta/schema/graphfile/upaNCE_generated.h"
#include "meta/schema/graphfile/software_generated.h"

namespace mv
{
    struct RuntimeModelNCE1Tensor
    {
        unsigned dimX;
        unsigned dimY;
        unsigned dimZ;
        unsigned strideX;
        unsigned strideY;
        unsigned strideZ;
        unsigned offset;
        unsigned location;
        unsigned datatype;
        unsigned order;
    };

    flatbuffers::Offset<Tensor> convertToFlatbuffer(RuntimeModelNCE1Tensor * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return MVCNN::CreateTensor(
            fbb,
            ref->dimX,
            ref->dimY,
            ref->dimZ,
            ref->strideX,
            ref->strideY,
            ref->strideZ,
            ref->offset,
            ref->location,
            ref->dataType,
            ref->order);
    }


    struct RuntimeModelNCE1FullyConnected : public RuntimeModelNCE1Layer
    {
        unsigned streamingMask;
        unsigned inputSize;
        unsigned outputSize;
        unsigned concatOffset;
        unsigned unloadCMX;
        unsigned overwriteInput;
        unsigned CMXSize_;
        unsigned reluSHVAcc_;
        unsigned shvNegSlope;
        unsigned shvPosSlope;
        unsigned desc_count;
        std::vector<unsigned> * descriptors;
        RuntimeModelNCE1Tensor * input;
        RuntimeModelNCE1Tensor * output;
        RuntimeModelNCE1Tensor * weight;
        RuntimeModelNCE1Tensor * bias;
    };

    flatbuffers::Offset<NCE1FCL> convertToFlatbuffer(RuntimeModelNCE1FullyConnected * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return MVCNN::CreateNCE1FCLDirect(
                    fbb,
                    ref->streamingMask,
                    ref->inputSize,
                    ref->outputSize,
                    ref->concatOffset,
                    ref->unloadCMX,
                    ref->overwriteInput,
                    ref->CMXSize,
                    ref->reluSHVAcc,
                    ref->shvNegSlope,
                    ref->shvPosSlope,
                    ref->desc_count,
                    ref->descriptors,
                    convertToFlatbuffer(ref->input, fbb),
                    convertToFlatbuffer(ref->output, fbb),
                    convertToFlatbuffer(ref->weight, fbb),
                    convertToFlatbuffer(ref->bias, fbb));
    }

    struct RuntimeModelNCE1Pool : public RuntimeModelNCE1Layer
    {
        unsigned streamingMask;
        unsigned inputSize;
        unsigned outputSize;
        unsigned concatOffset;
        unsigned unloadCMX;
        unsigned overwriteInput;
        unsigned CMXSize;
        unsigned reluSHVAcc;
        unsigned shvNegSlope;
        unsigned shvPosSlope;
        unsigned desc_count;
        std::vector<unsigned> descriptors;
        RuntimeModelNCE1Tensor * input;
        RuntimeModelNCE1Tensor * output;
        RuntimeModelNCE1Tensor * weight;
        RuntimeModelNCE1Tensor * bias;
    };

    flatbuffers::Offset<NCE1Pool> convertToFlatbuffer(RuntimeModelNCE1Pool * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return MVCNN::CreateNCE1PoolDirect(
                    fbb,
                    ref->streamingMask,
                    ref->inputSize,
                    ref->outputSize,
                    ref->concatOffset,
                    ref->unloadCMX,
                    ref->overwriteInput,
                    ref->CMXSize,
                    ref->reluSHVAcc,
                    ref->shvNegSlope,
                    ref->shvPosSlope,
                    ref->desc_count,
                    ref->descriptors,
                    convertToFlatbuffer(ref->input, fbb),
                    convertToFlatbuffer(ref->output, fbb),
                    convertToFlatbuffer(ref->weight, fbb),
                    convertToFlatbuffer(ref->bias, fbb));
    }

    struct RuntimeModelNCE1Conv : public RuntimeModelNCE1Layer
    {
        unsigned streamingMask;
        unsigned inputSize;
        unsigned outputSize;
        unsigned concatOffset;
        unsigned unloadCMX;
        unsigned overwriteInput;
        unsigned CMXSize;
        unsigned reluSHVAcc;
        unsigned shvNegSlope;
        unsigned shvPosSlope;
        unsigned descCount;
        std::vector<unsigned> descriptors;
        RuntimeModelNCE1Tensor * input;
        RuntimeModelNCE1Tensor * output;
        RuntimeModelNCE1Tensor * weight;
        RuntimeModelNCE1Tensor * bias;
    };

    flatbuffers::Offset<NCE1Conv> convertToFlatbuffer(RuntimeModelNCE1Conv * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return MVCNN::CreateNCE1ConvDirect(
                    fbb,
                    ref->streamingMask,
                    ref->inputSize,
                    ref->outputSize,
                    ref->concatOffset,
                    ref->unloadCMX,
                    ref->overwriteInput,
                    ref->CMXSize,
                    ref->reluSHVAcc,
                    ref->shvNegSlope,
                    ref->shvPosSlope,
                    ref->desc_count,
                    ref->descriptors,
                    convertToFlatbuffer(ref->input, fbb),
                    convertToFlatbuffer(ref->output, fbb),
                    convertToFlatbuffer(ref->weight, fbb),
                    convertToFlatbuffer(ref->bias, fbb));
    }

    struct RuntimeModelNCE1Layer
    {

    };

    flatbuffers::Offset<void> convertToFlatbuffer(RuntimeModelNCE1Layer * ref, RuntimeModelNCE1LayerType layerType, flatbuffers::FlatBufferBuilder& fbb)
    {
        switch (layerType)
        {
            case NONE:
                return convertToFlatbuffer((RuntimeModelNCE1Conv *) ref, flatbuffers::FlatBufferBuilder& fbb);
                break;
            case CONV:
                return convertToFlatbuffer((RuntimeModelNCE1Conv *) ref, flatbuffers::FlatBufferBuilder& fbb);
                break;
            case POOL:
                return convertToFlatbuffer((RuntimeModelNCE1Pool *) ref, flatbuffers::FlatBufferBuilder& fbb);
                break;
            case FULLYCONNECTED:
                return convertToFlatbuffer((RuntimeModelNCE1FullyConnected *) ref, flatbuffers::FlatBufferBuilder& fbb);
                break;
            default:
                break;
        }
    }

    enum RuntimeModelNCE1LayerType
    {
        NONE,
        CONV,
        POOL,
        FULLYCONNECTED
    };

    struct RuntimeModelNCE1Task : public RuntimeModelSpecificTask
    {
        RuntimeModelNCE1Layer * layer;
        RuntimeModelNCE1LayerType layerType;
    };

    flatbuffers::Offset<NCE1Task> convertToFlatbuffer(RuntimeModelNCE1Task * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return flatbuffers::Offset<NCE1Task> (
                    fbb,
                    ref->layer,
                    convertToFlatbuffer(ref, ref->layerType, fbb));
    }
}

#endif
