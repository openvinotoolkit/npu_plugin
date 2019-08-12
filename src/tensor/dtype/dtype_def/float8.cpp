#include "include/mcm/tensor/dtype/dtype_registry.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"
#include "include/mcm/base/exception/dtype_error.hpp"

namespace mv
{
    //Float 8 is handled with Integers
    MV_REGISTER_DTYPE(Float8)
    .setIsDoubleType(false)
    .setSizeInBits(8);
}
