#include "include/mcm/tensor/dtype/dtype_registry.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"


namespace mv
{

    // Float16 is actually treatead as Int
    MV_REGISTER_DTYPE(Float16)
    .setIsDoubleType(false)
    .setSizeInBits(16);
}
