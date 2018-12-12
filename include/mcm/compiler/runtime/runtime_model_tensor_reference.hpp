#ifndef MV_RUNTIME_MODEL_TENSOR_REFERENCE_
#define MV_RUNTIME_MODEL_TENSOR_REFERENCE_

#include <vector>
#include <cstdint>
#include "include/mcm/compiler/runtime/runtime_model_memory_location.hpp"
#include "include/mcm/compiler/runtime/runtime_model_dtypes.hpp"

namespace mv
{

    struct RuntimeModelTensorReference
    {
        std::vector<unsigned> dimensions;
        std::vector<unsigned> strides;
        unsigned leadingOffset;
        unsigned trailingOffset;
        unsigned dataIndex;
        unsigned sparsityIndex;
        RuntimeModelMemoryLocation * locale;
        RuntimeModelDType * dtype;
        unsigned quantScale;
        unsigned quantZero;
        unsigned quantShift;
    };
}


#endif //MV_RUNTIME_MODEL_TENSOR_REFERENCE_
