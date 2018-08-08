#ifndef MV_BLOB_MX_BCONV_HPP_
#define MV_BLOB_MX_BCONV_HPP_
#include "include/mcm/deployer/blob_serialization/bDefinition.hpp"
#include "include/mcm/deployer/blob_serialization/bTensor.hpp"
#include "include/mcm/utils/serializer/file_buffer.h"
#include "include/mcm/computation/tensor/tensor.hpp"
#include "include/mcm/computation/op/computation_op.hpp"
#include "include/mcm/deployer/blob_serialization/myriadX_hardware_descriptors.hpp"

namespace mv{
    class bConv2D : public Blob_Op_Definition
    {
        private:
            mv::Tensor input;
            mv::Tensor output;
            mv::Tensor taps;
            mv::dynamic_vector<float> bias;
            mv::Tensor scale;

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
            void writeStageInfo(WBuffer* b);
            bConv2D(mv::ComputationOp* it);
    };
}

#endif
