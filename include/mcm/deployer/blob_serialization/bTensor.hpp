#ifndef MV_BLOB_MX_BTENSOR_HPP_
#define MV_BLOB_MX_BTENSOR_HPP_

#include "include/mcm/computation/model/types.hpp"
#include "include/mcm/utils/serializer/file_buffer.h"

namespace mv
{
    class Blob_Tensor
    {
        public:
            uint32_t dimX;
            uint32_t dimY;
            uint32_t dimZ;
            uint32_t strideX;
            uint32_t strideY;
            uint32_t strideZ;
            uint32_t offset;
            uint32_t location;
            uint32_t dataType;
            uint32_t order;

            Blob_Tensor(int x, int y, int z,
                int sx, int sy, int sz,
                int offsetParam, int locationParam, int dtype, int orderParam);

            void write(WBuffer* b);
    };

}
#endif
