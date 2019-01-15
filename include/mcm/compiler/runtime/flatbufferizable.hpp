#ifndef MV_FLATBUFFERIZABLE_
#define MV_FLATBUFFERIZABLE_

#include "meta/schema/graphfile/graphfile_generated.h"

namespace mv
{
    struct Flatbufferizable
    {
        virtual flatbuffers::Offset<void> convertToFlatbuffer(flatbuffers::FlatBufferBuilder& fbb) = 0;
        virtual ~Flatbufferizable() = 0;
    };
}

#endif
