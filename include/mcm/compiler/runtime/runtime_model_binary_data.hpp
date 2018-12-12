#ifndef MV_RUNTIME_MODEL_BINARY_DATA_
#define MV_RUNTIME_MODEL_BINARY_DATA_

#include <vector>
#include <cstdint>
#include "include/mcm/compiler/runtime/runtime_model_dtypes.hpp"

namespace mv
{
    struct RuntimeModelBinaryData
    {
        RuntimeModelDType dType;
        char * data;
    };
}

#endif
