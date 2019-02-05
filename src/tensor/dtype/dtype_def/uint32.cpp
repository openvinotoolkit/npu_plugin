#include "include/mcm/tensor/dtype/dtype_registry.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"

namespace mv
{

    static std::function<BinaryData(const std::vector<double>&)> toBinaryFunc =
    [](const std::vector<double> & vals)->mv::BinaryData
    {
        std::vector<uint32_t> res(vals.begin(), vals.end());
        mv::BinaryData bdata;
        bdata.setU32(std::move(res));
        return bdata;
    };

    MV_REGISTER_DTYPE(UInt32)
    .setToBinaryFunc(toBinaryFunc)
    .setSizeInBytes(4);
}
