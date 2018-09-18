#ifndef MV_BLOB_MX_BDEF_HPP_
#define MV_BLOB_MX_BDEF_HPP_


#include "include/mcm/computation/op/op_type.hpp"

namespace mv
{
    class Blob_Op_Definition
    {
        public:
            uint32_t number_of_inputs;
            Blob_Op_Definition(OpType o);
            Blob_Op_Definition();
            static int getSerializedSize();

    };
}
#endif
