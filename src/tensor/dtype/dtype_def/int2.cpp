#include "include/mcm/tensor/dtype/dtype_registry.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"

namespace mv
{

    static std::function<BinaryData(const std::vector<double>&)> toBinaryFunc =
    [](const std::vector<double> & vals)->mv::BinaryData
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

        for(size_t i=0; i< vals.size(); i++)
        {
            if (i%4 == 0)
            {
                temp.data = 0;
                temp.b.llb = vals[i];
            }
            if (i%4 == 1)
                temp.b.lhb = vals[i];
            if (i%4 == 2)
                temp.b.hlb = vals[i];
            if (i%4 == 3)
            {
                temp.b.hhb = vals[i];
                res.push_back(temp.data);
            }
        }

        if (vals.size() % 4 != 0)
            res.push_back(temp.data);

        mv::BinaryData bdata(mv::DTypeType::Int2);
        bdata.setI2(std::move(res));
        return bdata;
    };

    MV_REGISTER_DTYPE(Int2).setToBinaryFunc(toBinaryFunc);
}
