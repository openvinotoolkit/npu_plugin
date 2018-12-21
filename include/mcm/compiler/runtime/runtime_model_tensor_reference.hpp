#ifndef MV_RUNTIME_MODEL_TENSOR_REFERENCE_
#define MV_RUNTIME_MODEL_TENSOR_REFERENCE_

#include <vector>
#include <cstdint>
#include "include/mcm/compiler/runtime/runtime_model_memory_location.hpp"
#include "include/mcm/compiler/runtime/runtime_model_dtypes.hpp"
#include "meta/schema/graphfile/graphfile_generated.h"
#include "meta/schema/graphfile/memoryManagement_generated.h"


namespace mv
{

    struct RuntimeModelTensorReference
    {
        std::vector<unsigned> * dimensions;
        std::vector<unsigned> * strides;
        unsigned leadingOffset;
        unsigned trailingOffset;
        unsigned dataIndex;
        unsigned sparsityIndex;
        RuntimeModelMemoryLocation locale;
        RuntimeModelDType dtype;
        std::vector<signed char> * quantScale;
        std::vector<signed char> * quantZero;
        std::vector<signed char> * quantShift;
    };


    flatbuffers::Offset<MVCNN::TensorReference> convertToFlatbuffer(RuntimeModelTensorReference * ref, flatbuffers::FlatBufferBuilder& fbb)
    {

        auto dataReference = MVCNN::CreateIndirectDataReference(fbb, ref->dataIndex, ref->sparsityIndex);

        return CreateTensorReferenceDirect(fbb,
                                    ref->dimensions,
                                    ref->strides,
                                    ref->leadingOffset,
                                    ref->trailingOffset,
                                    dataReference,
                                    static_cast<MVCNN::MemoryLocation>(ref->locale),
                                    static_cast<MVCNN::DType>(ref->dtype),
                                    ref->quantScale,
                                    ref->quantZero,
                                    ref->quantShift);
    }

    std::vector<flatbuffers::Offset<MVCNN::TensorReference>> * convertToFlatbuffer(std::vector<RuntimeModelTensorReference*> * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        std::vector<flatbuffers::Offset<MVCNN::TensorReference>> * toReturn = new std::vector<flatbuffers::Offset<MVCNN::TensorReference>>();
        for(unsigned i = 0; i < ref->size(); ++i)
        {
            RuntimeModelTensorReference * currentRef = ref->at(i);
            flatbuffers::Offset<MVCNN::TensorReference> currentOffset = convertToFlatbuffer(currentRef, fbb);
            toReturn->push_back(currentOffset);
        }
        return toReturn;
    }
}


#endif //MV_RUNTIME_MODEL_TENSOR_REFERENCE_
