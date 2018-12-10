#ifndef MV_RUNTIME_MODEL_HEADER_
#define MV_RUNTIME_MODEL_HEADER_

#include <cstdint>
#include <string>
#include "include/mcm/compiler/runtime/runtime_model_tensor.hpp"
#include "include/mcm/compiler/runtime/runtime_model_link.hpp"

namespace mv
{
    struct RuntimeModelHeader
    {
        //Version information
        unsigned majorV_;
        unsigned minorV_;
        unsigned patchV_;
        std::string hash_;

        //Inputs and outputs
        std::vector<RuntimeModelTensorReference> netInput_;
        std::vector<RuntimeModelTensorReference> netOutput_;

        unsigned taskCount_;
        unsigned layerCount_;

        //Resources
        unsigned shaveMask_;
        unsigned nce1Mask_;
        unsigned dpuMask_;
        unsigned leonCmx_;
        unsigned nnCmx_;
        unsigned ddrScratch_;

        //Network structure
        std::vector<Link> links_;
        std::vector<unsigned> firstID_;
    };
}

#endif
