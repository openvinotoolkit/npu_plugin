#ifndef MV_RUNTIME_MODEL_BARRIER_
#define MV_RUNTIME_MODEL_BARRIER_

#include <vector>
#include "KeemBayFBSchema/compiledSchemas/structure_generated.h"

namespace mv
{
    struct RuntimeModelBarrier
    {
        unsigned barrierID;
        unsigned consumerCount;
        unsigned producerCount;
    };

    struct RuntimeModelBarrierReference
    {
        unsigned waitBarrier;
        std::vector<unsigned> * updateBarriers;
    };


    flatbuffers::Offset<MVCNN::Barrier> convertToFlatbuffer(RuntimeModelBarrier * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return MVCNN::CreateBarrier(fbb, ref->barrierID, ref->consumerCount, ref->producerCount);
    }

    flatbuffers::Offset<MVCNN::BarrierReference> convertToFlatbuffer(RuntimeModelBarrierReference * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return MVCNN::CreateBarrierReferenceDirect(fbb, ref->waitBarrier, ref->updateBarriers);
    }

    std::vector<flatbuffers::Offset<MVCNN::Barrier>> * convertToFlatbuffer(std::vector<RuntimeModelBarrier*> * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        std::vector<flatbuffers::Offset<MVCNN::Barrier>> * toReturn = new std::vector<flatbuffers::Offset<MVCNN::Barrier>>();
        for(unsigned i = 0; i < ref->size(); ++i)
        {
            RuntimeModelBarrier * currentRef = ref->at(i);
            flatbuffers::Offset<MVCNN::Barrier> currentOffset = convertToFlatbuffer(currentRef, fbb);
            toReturn->push_back(currentOffset);
        }
        return toReturn;
    }

}

#endif
