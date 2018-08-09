#ifndef MV_BLOB_MX_BCONV_HPP_
#define MV_BLOB_MX_BCONV_HPP_
#include "include/mcm/deployer/blob_serialization/bDefinition.hpp"
#include "include/mcm/deployer/blob_serialization/bTensor.hpp"
#include "include/mcm/deployer/blob_serialization/myriadX_hardware_descriptors.hpp"
#include "include/mcm/utils/serializer/file_buffer.h"
#include "include/mcm/computation/tensor/tensor.hpp"
#include "include/mcm/computation/op/computation_op.hpp"
#include "include/mcm/computation/model/op_model.hpp"

namespace mv{

    class Blob_buffer;  // Forward Declaration

    class bConv2D : public Blob_Op_Definition
    {
        private:
            mv::Data::TensorIterator input;
            mv::Data::TensorIterator output;
            mv::Data::TensorIterator taps;
            mv::dynamic_vector<float> bias;
            mv::Data::TensorIterator scale;

            // Hardware Fields
            uint32_t opMode;
            uint32_t streamingMask;
            uint32_t concatOffset;
            uint32_t unloadCMX;
            uint32_t overwriteInput;
            uint32_t CMXSize;
            uint32_t reluSHVAcc;
            uint32_t shvNegSlope;
            uint32_t shvPosSlope;
            uint32_t desc_count;

            cnnConvolutionPoolStructure * descriptors;

        public:
            uint32_t number_of_inputs = 2;
            void writeStageInfo(mv::OpModel * om, mv::Blob_buffer* b);
            bConv2D(mv::ComputationOp* it);
    };
}

#endif
