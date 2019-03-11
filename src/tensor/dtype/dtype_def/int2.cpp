#include "include/mcm/tensor/dtype/dtype_registry.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"

namespace mv
{

    static std::function<BinaryData(const std::vector<DataElement>&)> toBinaryFunc =
    [](const std::vector<DataElement> & vals)->mv::BinaryData
    {
        std::vector<int8_t> res;

        union
        {
            struct bits {
                uint8_t llb : 2;
                uint8_t lhb : 2;
                uint8_t hlb : 2;
                uint8_t hhb : 2;
            } b;
            uint8_t data;
        } temp;

        int64_t val;
        for(size_t i=0; i< vals.size(); i++)
        {
            val = vals[i];
            if (i%4 == 0)
            {
                temp.data = 0;
                temp.b.llb = val;
            }
            if (i%4 == 1)
                temp.b.lhb = val;
            if (i%4 == 2)
                temp.b.hlb = val;
            if (i%4 == 3)
            {
                temp.b.hhb = val;
                res.push_back(temp.data);
            }
        }

        if (vals.size() % 4 != 0)
            res.push_back(temp.data);

        mv::BinaryData bdata;
        bdata.setI2(std::move(res));
        return bdata;
    };

    MV_REGISTER_DTYPE(Int2)
    .setToBinaryFunc(toBinaryFunc)
    .setIsDoubleType(false)
    .setSizeInBits(2);
}
