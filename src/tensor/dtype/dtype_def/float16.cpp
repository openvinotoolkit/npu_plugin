#include "include/mcm/tensor/dtype/dtype_registry.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"


namespace mv
{

    static std::function<BinaryData(const std::vector<DataElement>&)> toBinaryFunc =
    [](const std::vector<DataElement> & vals)->mv::BinaryData
    {
        std::vector<int16_t> res;
        mv_num_convert cvtr;
        for_each(vals.begin(), vals.end(), [&](double  val)
        {
            res.push_back(cvtr.fp32_to_fp16(val));
        });
        mv::BinaryData bdata;
        bdata.setFp16(std::move(res));
        return bdata;
    };

    MV_REGISTER_DTYPE(Float16)
    .setToBinaryFunc(toBinaryFunc)
    .setIsDoubleType(true)
    .setSizeInBits(16);
}
