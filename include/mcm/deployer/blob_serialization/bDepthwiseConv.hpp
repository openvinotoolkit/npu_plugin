#ifndef MV_BLOB_MX_BDEPTHWISECONV_HPP_
#define MV_BLOB_MX_BDEPTHWISECONV_HPP_

#include "include/mcm/deployer/blob_serialization/bDefinition.hpp"
#include "include/mcm/deployer/blob_serialization/bTensor.hpp"
#include "include/mcm/deployer/blob_serialization/myriadX_hardware_descriptors.hpp"
#include "include/mcm/utils/serializer/file_buffer.hpp"
#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/computation/op/computation_op.hpp"
#include "include/mcm/computation/model/op_model.hpp"

namespace mv{

    class Blob_buffer;  // Forward Declaration

    class bDepthwiseConv2D : public Blob_Op_Definition
    {
        private:
            mv::Data::TensorIterator input;
            mv::Data::TensorIterator output;
            mv::Data::TensorIterator taps;
            mv::Data::TensorIterator bias;
            mv::Data::TensorIterator scale;

            std::string bias_name;
            std::string scale_name;

            // Software Fields
            uint32_t radixX;
            uint32_t radixY;
            uint32_t strideX;
            uint32_t strideY;
            uint32_t padX;
            uint32_t padY;
            uint32_t padStyle;
            uint32_t dilation;

        public:
            uint32_t number_of_inputs = 2;
            void writeStageInfo(OpModel& om, mv::Blob_buffer* b);
            bDepthwiseConv2D(mv::Control::OpListIterator it);
            ~bDepthwiseConv2D();
    };
}

#endif //MV_BLOB_MX_BDEPTHWISECONV_HPP_
