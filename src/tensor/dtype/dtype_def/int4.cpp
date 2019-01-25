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
                uint8_t lb : 4;
                uint8_t hb : 4;
            } b;
            uint8_t data;
        } temp;

        for(size_t i=0; i< vals.size(); i++)
        {
            if (i%2 == 0)
            {
                temp.b.hb = 0;
                temp.b.lb = vals[i];
            }
            if (i%2 == 1)
            {
                temp.b.hb = vals[i];
                res.push_back(temp.data);
            }
        }

        if (vals.size() % 2 != 0)
            res.push_back(temp.data);

        mv::BinaryData bdata;
        bdata.setI4(std::move(res));
        return bdata;
    };

    MV_REGISTER_DTYPE(Int4).setToBinaryFunc(toBinaryFunc);
}
