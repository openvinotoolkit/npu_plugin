#include "include/mcm/tensor/dtype/dtype_registry.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"
#include "include/mcm/base/exception/dtype_error.hpp"

namespace mv
{

    static std::function<BinaryData(const std::vector<double>&)> toBinaryFunc =
    [](const std::vector<double> & vals)->mv::BinaryData
    {
        (void) vals;
        throw DTypeError("DType", "conversion for Float8 is not supported yet");
    };

    MV_REGISTER_DTYPE("Float8").setToBinaryFunc(toBinaryFunc);
}
