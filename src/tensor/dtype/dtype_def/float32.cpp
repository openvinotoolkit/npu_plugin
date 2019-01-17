#include "include/mcm/tensor/dtype/dtype_registry.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"

namespace mv
{

    static std::function<BinaryData(const std::vector<double>&)> toBinaryFunc =
    [](const std::vector<double> & vals)->mv::BinaryData
    {
        std::vector<float> res(vals.begin(), vals.end());
        mv::BinaryData bdata(mv::DTypeType::Float32);
        bdata.setFp32(std::move(res));
        return bdata;
    };

    MV_REGISTER_DTYPE("Float32").setToBinaryFunc(toBinaryFunc);
}
