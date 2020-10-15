#include "include/mcm/tensor/dtype/dtype_registry.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"

namespace mv
{
    MV_REGISTER_DTYPE(UInt64)
    .setIsDoubleType(false)
    .setIsSigned(false)
    .setSizeInBits(64);
}
