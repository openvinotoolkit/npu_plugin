#ifndef MV_RUNTIME_MODEL_LINK_
#define MV_RUNTIME_MODEL_LINK_

#include <string>
#include <vector>

namespace mv
{
    struct Link
    {
        unsigned thisId_;
        std::string name_;
        std::vector<unsigned> sourceID_;
        std::vector<unsigned> sinkID_;
    };
}

#endif
