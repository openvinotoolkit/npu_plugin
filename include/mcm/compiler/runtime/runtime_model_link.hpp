#ifndef MV_RUNTIME_MODEL_LINK_
#define MV_RUNTIME_MODEL_LINK_

#include <string>
#include <vector>

namespace mv
{
    struct Link
    {
        unsigned thisId;
        std::string name;
        std::vector<unsigned> sourceID;
        std::vector<unsigned> sinkID;
    };
}

#endif
