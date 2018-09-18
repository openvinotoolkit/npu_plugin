#ifndef MV_BLOB_MX_BCOMPAT_HPP_
#define MV_BLOB_MX_BCOMPAT_HPP_

#include "include/mcm/deployer/blob_serialization/bDefinition.hpp"
#include "include/mcm/deployer/blob_serialization/bTensor.hpp"
#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/computation/op/computation_op.hpp"
#include "include/mcm/computation/model/op_model.hpp"
// #include "include/mcm/deployer/blob_serialization/blob_serializer.hpp"

namespace mv{

    class Blob_buffer; // Forward Declaration

    class bCompatibility : public Blob_Op_Definition
    {
        private:
            mv::Data::TensorIterator input;
            mv::Data::TensorIterator output;
        public:
            uint32_t number_of_inputs = 1;
            void writeStageInfo(mv::OpModel * om, Blob_buffer* b);
            bCompatibility(mv::ComputationOp* it);
            static int getSerializedSize();
    };
}
#endif
