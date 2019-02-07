#include "include/mcm/tensor/dtype/dtype_registry.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"

namespace mv
{

    static std::function<BinaryData(const std::vector<double>&)> toBinaryFunc =
    [](const std::vector<double> & vals)->mv::BinaryData
    {
        mv::BinaryData bdata;
        bdata.setFp64(vals);
        return bdata;
    };

    MV_REGISTER_DTYPE(Float64)
    .setToBinaryFunc(toBinaryFunc)
    .setSizeInBits(64);
}
