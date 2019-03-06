#include "include/mcm/tensor/dtype/dtype_registry.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"

namespace mv
{

    static std::function<BinaryData(const std::vector<DataElement>&)> toBinaryFunc =
    [](const std::vector<DataElement> & vals)->mv::BinaryData
    {
        mv::BinaryData bdata;
        std::vector<double> res(vals.begin(), vals.end());
        bdata.setFp64(res);
        return bdata;
    };

    MV_REGISTER_DTYPE(Float64)
    .setToBinaryFunc(toBinaryFunc)
    .setIsDoubleType(true)
    .setSizeInBits(64);
}
