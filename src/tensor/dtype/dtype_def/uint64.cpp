#include "include/mcm/tensor/dtype/dtype_registry.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"

namespace mv
{

    static std::function<BinaryData(const std::vector<double>&)> toBinaryFunc =
    [](const std::vector<double> & vals)->mv::BinaryData
    {
        std::vector<uint64_t> res(vals.begin(), vals.end());
        mv::BinaryData bdata;
        bdata.setU64(std::move(res));
        return bdata;
    };

    MV_REGISTER_DTYPE(UInt64)
    .setToBinaryFunc(toBinaryFunc)
    .setSizeInBits(64);
}
