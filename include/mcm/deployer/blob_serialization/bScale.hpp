#ifndef MV_BLOB_MX_BSCALE_HPP_
#define MV_BLOB_MX_BSCALE_HPP_
#include "include/mcm/deployer/blob_serialization/bDefinition.hpp"
#include "include/mcm/deployer/blob_serialization/bTensor.hpp"
#include "include/mcm/deployer/blob_serialization/myriadX_hardware_descriptors.hpp"
#include "include/mcm/utils/serializer/file_buffer.hpp"
#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/computation/op/computation_op.hpp"
#include "include/mcm/computation/model/op_model.hpp"

namespace mv{

    class Blob_buffer;  // Forward Declaration

    class bScale : public Blob_Op_Definition
    {
        private:
            mv::Data::TensorIterator input;
            mv::Data::TensorIterator output;
            mv::Data::TensorIterator taps;
            mv::Data::TensorIterator bias;

            std::string bias_name;

        public:
            uint32_t number_of_inputs = 1;
            void writeStageInfo(mv::OpModel * om, mv::Blob_buffer* b);
            bScale(mv::ComputationOp* it);
    };
}

#endif
