#ifndef MV_BLOB_MX_BPRELU_HPP_
#define MV_BLOB_MX_BPRELU_HPP_

#include "include/mcm/deployer/blob_serialization/bDefinition.hpp"
#include "include/mcm/deployer/blob_serialization/bTensor.hpp"
#include "include/mcm/computation/tensor/tensor.hpp"
#include "include/mcm/computation/op/computation_op.hpp"
#include "include/mcm/computation/model/op_model.hpp"
// #include "include/mcm/deployer/blob_serialization/blob_serializer.hpp"

namespace mv{

    class Blob_buffer; // Forward Declaration

    class bPRelu : public Blob_Op_Definition
    {
        private:
            mv::Data::TensorIterator input;
            mv::Data::TensorIterator output;
            mv::Data::TensorIterator neg_slope;
        public:
            uint32_t number_of_inputs = 1;
            void writeStageInfo(mv::OpModel * om, Blob_buffer* b);
            bPRelu(mv::ComputationOp* it);
            static int getSerializedSize();
    };
}
#endif
