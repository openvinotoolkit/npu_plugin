#include "include/mcm/tensor/dtype/dtype_registry.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"

namespace mv
{

    static std::function<BinaryData(const std::vector<DataElement>&)> toBinaryFunc =
    [](const std::vector<DataElement> & vals)->mv::BinaryData
    {
        std::vector<float> res(vals.begin(), vals.end());
        mv::BinaryData bdata;
        bdata.setFp32(std::move(res));
        return bdata;
    };

    MV_REGISTER_DTYPE(Float32)
    .setToBinaryFunc(toBinaryFunc)
    .setIsDoubleType(true)
    .setSizeInBits(32);
}
