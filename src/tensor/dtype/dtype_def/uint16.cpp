#include "include/mcm/tensor/dtype/dtype_registry.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"

namespace mv
{
    MV_REGISTER_DTYPE(UInt16)
    .setIsDoubleType(false)
    .setSizeInBits(16);
}
