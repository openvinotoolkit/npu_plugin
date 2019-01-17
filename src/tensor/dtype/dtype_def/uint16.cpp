#include "include/mcm/tensor/dtype/dtype_registry.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"

namespace mv
{

    static std::function<BinaryData(const std::vector<double>&)> toBinaryFunc =
    [](const std::vector<double> & vals)->mv::BinaryData
    {
        std::vector<uint16_t> res(vals.begin(), vals.end());
        mv::BinaryData bdata("UInt16");
        bdata.setU16(std::move(res));
        return bdata;
    };

    MV_REGISTER_DTYPE(UInt16).setToBinaryFunc(toBinaryFunc);
}
