#ifndef MV_RUNTIME_MODEL_LINK_
#define MV_RUNTIME_MODEL_LINK_

#include <string>
#include <vector>

#include "KeemBayFBSchema/compiledSchemas/structure_generated.h"

namespace mv
{
    struct RuntimeModelLink
    {
        unsigned thisId;
        std::string name;
        std::vector<unsigned> * sourceID;
        std::vector<unsigned> * sinkID;
    };

    flatbuffers::Offset<MVCNN::Link> convertToFlatbuffer(RuntimeModelLink * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        return MVCNN::CreateLinkDirect(fbb,
                ref->thisId,
                ref->name.c_str(),
                ref->sourceID,
                ref->sinkID);
    }

    std::vector<flatbuffers::Offset<MVCNN::Link>> * convertToFlatbuffer(std::vector<RuntimeModelLink*> * ref, flatbuffers::FlatBufferBuilder& fbb)
    {
        std::vector<flatbuffers::Offset<MVCNN::Link>> * toReturn = new std::vector<flatbuffers::Offset<MVCNN::Link>>();
        for(unsigned i = 0; i < ref->size(); ++i)
        {
            RuntimeModelLink * currentRef = ref->at(i);
            flatbuffers::Offset<MVCNN::Link> currentOffset = convertToFlatbuffer(currentRef, fbb);
            toReturn->push_back(currentOffset);
        }
        return toReturn;
    }

}

#endif
