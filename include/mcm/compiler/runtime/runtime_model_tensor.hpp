#ifndef MV_RUNTIME_MODEL_TENSOR_
#define MV_RUNTIME_MODEL_TENSOR_

#include <vector>
#include <cstdint>
#include "include/mcm/compiler/runtime/runtime_model_memory.hpp"
#include "include/mcm/compiler/runtime/runtime_model_dtypes.hpp"

namespace mv
{

    struct RuntimeModelTensorReference
    {
        std::vector<unsigned> dimensions_;
        std::vector<unsigned> strides_;
        unsigned leadingOffset_;
        unsigned trailingOffset_;
        unsigned dataIndex_;
        unsigned sparsityIndex_;
        RuntimeModelMemoryLocation locale_;
        RuntimeModelDType dtype_;
        unsigned quantScale_;
        unsigned quantZero_;
        unsigned quantShift_;
    };
}


#endif
