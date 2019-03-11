#include "include/mcm/tensor/dtype/dtype_registry.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"

namespace mv
{

    static std::function<BinaryData(const std::vector<DataElement>&)> toBinaryFunc =
    [](const std::vector<DataElement> & vals)->mv::BinaryData
    {
        std::vector<uint16_t> res;
        for_each(vals.begin(), vals.end(), [&](int64_t  val)
        {
            res.push_back(val);
        });
        mv::BinaryData bdata;
        bdata.setU16(std::move(res));
        return bdata;
    };

    MV_REGISTER_DTYPE(UInt16)
    .setToBinaryFunc(toBinaryFunc)
    .setIsDoubleType(false)
    .setSizeInBits(16);
}
