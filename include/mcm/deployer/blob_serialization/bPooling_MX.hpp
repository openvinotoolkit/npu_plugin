#ifndef MV_BLOB_MX_BPOOL_HPP_
#define MV_BLOB_MX_BPOOL_HPP_
#include "include/mcm/deployer/blob_serialization/bDefinition.hpp"
#include "include/mcm/deployer/blob_serialization/bTensor.hpp"
#include "include/mcm/deployer/blob_serialization/myriadX_hardware_descriptors.hpp"
#include "include/mcm/utils/serializer/file_buffer.h"
#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/computation/op/computation_op.hpp"
#include "include/mcm/computation/model/op_model.hpp"

namespace mv{

    class Blob_buffer;  // Forward Declaration

    class bPooling_MX : public Blob_Op_Definition
    {
        private:
        mv::Data::TensorIterator input;
        mv::Data::TensorIterator output;

        std::string bias_name;
        std::string scale_name;

        // Hardware Fields
        std::vector<size_t> DPUmodeVector;
        std::vector<size_t> input_lines_processed;
        std::vector<size_t> output_lines_processed;
        std::vector<size_t> input_line_start;
        std::vector<size_t> output_line_start;
        std::vector<size_t> outputChannelPerformed;
        size_t splits_over_H;
        size_t splits_over_iC;
        size_t inputChannels;
        size_t inputWidthPadded;
        size_t outputWidthPadded;

        uint32_t streamingMask;
        uint32_t concatOffset;
        uint32_t unloadCMX;
        uint32_t overwriteInput;
        uint32_t CMXSize;
        uint32_t reluSHVAcc;
        uint32_t shvNegSlope;
        uint32_t shvPosSlope;
        uint32_t desc_count;



        // Software Fields
        uint32_t radixX;
        uint32_t radixY;
        uint32_t strideX;
        uint32_t strideY;
        uint32_t padX;
        uint32_t padY;
        uint32_t padStyle;
        uint32_t dilation;

        bool NCE1_Compatible;

        cnnConvolutionPoolStructure * descriptors;

        public:
            uint32_t number_of_inputs = 1;
            void writeStageInfo(mv::OpModel * om, mv::Blob_buffer* b);
            bPooling_MX(mv::ComputationOp* it);
            static int getSerializedSize(int platform);

    };
}

#endif
