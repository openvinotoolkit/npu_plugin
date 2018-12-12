#ifndef MV_RUNTIME_MODEL_HEADER_
#define MV_RUNTIME_MODEL_HEADER_

#include <cstdint>
#include <string>
#include "include/mcm/compiler/runtime/runtime_model_tensor_reference.hpp"
#include "include/mcm/compiler/runtime/runtime_model_link.hpp"

namespace mv
{
    struct RuntimeModelHeader
    {
        //Version information
        unsigned majorV;
        unsigned minorV;
        unsigned patchV;
        std::string hash;

        //Inputs and outputs
        std::vector<*RuntimeModelTensorReference> netInput;
        std::vector<*RuntimeModelTensorReference> netOutput;

        unsigned taskCount;
        unsigned layerCount;

        //Resources
        unsigned shaveMask;
        unsigned nce1Mask;
        unsigned dpuMask;
        unsigned leonCmx;
        unsigned nnCmx;
        unsigned ddrScratch;

        //Network structure
        std::vector<*Link> links;
        std::vector<unsigned> firstID;
    };
}

#endif
