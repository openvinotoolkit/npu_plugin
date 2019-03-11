#include "include/mcm/tensor/dtype/dtype_registry.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"

namespace mv
{

    static std::function<BinaryData(const std::vector<DataElement>&)> toBinaryFunc =
    [](const std::vector<DataElement> & vals)->mv::BinaryData
    {
        std::vector<int64_t> res(vals.begin(), vals.end());
        mv::BinaryData bdata;
        bdata.setI64(std::move(res));
        return bdata;
    };

    MV_REGISTER_DTYPE(Int64)
    .setToBinaryFunc(toBinaryFunc)
    .setIsDoubleType(false)
    .setSizeInBits(64);
}
