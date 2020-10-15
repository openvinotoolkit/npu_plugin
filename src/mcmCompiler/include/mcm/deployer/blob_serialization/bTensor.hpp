#ifndef MV_BLOB_MX_BTENSOR_HPP_
#define MV_BLOB_MX_BTENSOR_HPP_

#include "include/mcm/utils/serializer/file_buffer.hpp"
#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/deployer/blob_serialization/bRelocation.hpp"
#include "include/mcm/deployer/blob_serialization/blob_serializer.hpp"

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
            std::string allocator_name;

            Blob_Tensor(int x, int y, int z,
                int sx, int sy, int sz,
                int offsetParam, int locationParam, int dtype, int orderParam);

            Blob_Tensor(mv::DataModel& dm, mv::ControlModel& cm, mv::RelocationTable& rt, mv::Data::TensorIterator t);

            void write(WBuffer* b);
    };

}
#endif
