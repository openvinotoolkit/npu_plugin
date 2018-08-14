#ifndef MV_BLOB_MX_BPOOL_HPP_
#define MV_BLOB_MX_BPOOL_HPP_

#include "include/mcm/deployer/blob_serialization/bDefinition.hpp"
#include "include/mcm/deployer/blob_serialization/bTensor.hpp"
#include "include/mcm/computation/tensor/tensor.hpp"
#include "include/mcm/computation/op/computation_op.hpp"
#include "include/mcm/computation/model/op_model.hpp"
// #include "include/mcm/deployer/blob_serialization/blob_serializer.hpp"

namespace mv{

    class Blob_buffer; // Forward Declaration

    class bPooling : public Blob_Op_Definition
    {
        private:
            int kernelRadixX;
            int kernelRadixY;
            int kernelStrideX;
            int kernelStrideY;
            int kernelPadX;
            int kernelPadY;
            int kernelPadStyle;

            mv::Data::TensorIterator input;
            mv::Data::TensorIterator output;
        public:
            uint32_t number_of_inputs = 1;
            void writeStageInfo(mv::OpModel * om, Blob_buffer* b);
            bPooling(mv::ComputationOp* it);
    };
}
#endif
