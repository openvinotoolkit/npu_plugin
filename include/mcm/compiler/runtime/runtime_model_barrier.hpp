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

    std::vector<flatbuffers::Offset<MVCNN::Barrier>> convertToFlatbuffer(std::vector<RuntimeModelBarrier> * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        std::vector<flatbuffers::Offset<MVCNN::Barrier>> toReturn;
        for(unsigned i = 0; i < ref->size(); ++i)
            toReturn.push_back(convertToFlatbuffer(ref->at(i), fbb));
        return toReturn;
    }

    flatbuffers::Offset<MVCNN::BarrierReference> convertToFlatbuffer(RuntimeModelBarrierReference * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return MVCNN::CreateBarrierReferenceDirect(fbb, ref->waitBarrier, ref->updateBarriers);
    }

}

#endif
