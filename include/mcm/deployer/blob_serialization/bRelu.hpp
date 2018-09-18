#ifndef MV_BLOB_MX_BSRELU_HPP_
#define MV_BLOB_MX_BSRELU_HPP_

#include "include/mcm/deployer/blob_serialization/bDefinition.hpp"
#include "include/mcm/deployer/blob_serialization/bTensor.hpp"
#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/computation/op/computation_op.hpp"
#include "include/mcm/computation/model/op_model.hpp"
// #include "include/mcm/deployer/blob_serialization/blob_serializer.hpp"

namespace mv{

    class Blob_buffer; // Forward Declaration

    class bRelu : public Blob_Op_Definition
    {
        private:
            int opX;
            int post_strideX;
            int post_strideY;

            mv::Data::TensorIterator input;
            mv::Data::TensorIterator output;
        public:
            uint32_t number_of_inputs = 1;
            void writeStageInfo(mv::OpModel * om, Blob_buffer* b);
            bRelu(mv::ComputationOp* it);
            static int getSerializedSize();
    };
}
#endif
