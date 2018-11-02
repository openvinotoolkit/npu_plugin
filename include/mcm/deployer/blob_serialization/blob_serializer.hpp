#ifndef MV_BLOB_SERIALIZER_HPP_
#define MV_BLOB_SERIALIZER_HPP_

/**
* serializer.hpp contains classes that output to file compute graph representations in various formats.
*
* @author Patrick Doyle, Ian Hunter
* @date 4/27/2018
*/
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "include/mcm/utils/serializer/file_buffer.h"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/deployer/blob_serialization/myriadX_hardware_descriptors.hpp"
#include "include/mcm/deployer/blob_serialization/bDefinition.hpp"
#include "include/mcm/deployer/blob_serialization/bTensor.hpp"
#include "include/mcm/deployer/blob_serialization/bConv_MX.hpp"
#include "include/mcm/deployer/blob_serialization/bDepthwiseConv.hpp"
#include "include/mcm/deployer/blob_serialization/bRelocation.hpp"
#include "include/mcm/deployer/blob_serialization/bPooling_MX.hpp"
#include "include/mcm/deployer/blob_serialization/bSoftmax.hpp"
#include "include/mcm/deployer/blob_serialization/bRelu.hpp"
#include "include/mcm/deployer/blob_serialization/bPRelu.hpp"
#include "include/mcm/deployer/blob_serialization/bScale.hpp"
#include "include/mcm/deployer/blob_serialization/bEltwise.hpp"
#include "include/mcm/deployer/blob_serialization/bInnerProduct.hpp"
#include "include/mcm/deployer/blob_serialization/bCompatibility.hpp"
#include <assert.h>

#define BLOB_VERSION_MAJOR 2
#define BLOB_VERSION_MINOR 3
#define BLOB_MAGIC_NUMBER 8708

#define BLOB_DEFAULT_IMPLEMENTATION 0x80000000

namespace mv
{

    class Blob_stage
    {
        public:
            std::size_t next ;
            std::size_t op_type ;
            std::size_t implementation  ;
            std::size_t preop_type  ;
            std::size_t postop_type ;

            std::size_t radixX;
            std::size_t radixY;
            std::size_t radixStrideX;
            std::size_t radixStrideY;
            std::size_t padX;
            std::size_t padY;
            std::size_t padStyle;
            std::size_t dilation;

            std::size_t InputDimX;
            std::size_t InputDimY;
            std::size_t InputDimZ;
            std::size_t InputStrideX;
            std::size_t InputStrideY;
            std::size_t InputStrideZ;
            std::size_t InputOffset;
            std::size_t InputLocation;
            std::size_t InputDataType;
            std::size_t InputOrder;
            std::size_t Input1Offset;
            std::size_t Input1Location;
            std::size_t TBOffset;

            std::size_t OutputDimX;
            std::size_t OutputDimY;
            std::size_t OutputDimZ;
            std::size_t OutputStrideX;
            std::size_t OutputStrideY;
            std::size_t OutputStrideZ;
            std::size_t OutputOffset;
            std::size_t OutputLocation;
            std::size_t OutputDataType;
            std::size_t OutputOrder;

            std::size_t TapsDimX;
            std::size_t TapsDimY;
            std::size_t TapsDimZ;
            std::size_t TapsStrideX;
            std::size_t TapsStrideY;
            std::size_t TapsStrideZ;
            std::size_t TapsOffset;
            std::size_t TapsLocation;
            std::size_t TapsDataType;
            std::size_t TapsOrder;

            std::size_t BiasDimX;
            std::size_t BiasDimY;
            std::size_t BiasDimZ;
            std::size_t BiasStrideX;
            std::size_t BiasStrideY;
            std::size_t BiasStrideZ;
            std::size_t BiasOffset;
            std::size_t BiasLocation;
            std::size_t BiasDataType;
            std::size_t BiasOrder;

            Blob_stage()
            {
                next = 0x0000 ;
                op_type = 0x0000;
                implementation = 0x80000000 ;

                radixX = 3 ;
                radixY = 3 ;
                radixStrideX = 2 ;
                radixStrideY = 2 ;
                padX = 0 ;
                padY = 0 ;
                padStyle = 2 ;
                dilation = 1 ;

                InputDimX = 32 ;
                InputDimY = 32 ;
                InputDimZ = 3 ;
                InputStrideX = 2 ;
                InputStrideY = 64 ;
                InputStrideZ = 2 ;
                InputOffset = 0 ;
                InputLocation = 1 ;
                Input1Offset = 0;
                Input1Location = 1;
                InputDataType = 0 ;
                InputOrder = 0 ;

                OutputDimX = 16 ;
                OutputDimY = 16 ;
                OutputDimZ = 8 ;
                OutputStrideX = 2 ;
                OutputStrideY = 0x10 ;
                OutputStrideZ = 2 ;
                OutputOffset = 0 ;
                OutputLocation = 2 ;
                OutputDataType = 0 ;
                OutputOrder = 0 ;

                TapsDimX = 9 ;
                TapsDimY = 1 ;
                TapsDimZ = 1 ;
                TapsStrideX = 2 ;
                TapsStrideY = 2 ;
                TapsStrideZ = 2 ;
                TapsOffset = 0 ;
                TBOffset = 0 ;
                TapsLocation = 3 ;
                TapsDataType = 0 ;
                TapsOrder = 3 ;

                BiasDimX = 64 ;
                BiasDimY = 1 ;
                BiasDimZ = 1 ;
                BiasStrideX = 2 ;
                BiasStrideY = 128 ;
                BiasStrideZ = 128 ;
                BiasOffset = 0 ;
                BiasLocation = 3 ;
                BiasDataType = 0 ;
                BiasOrder = 1 ;

                preop_type = 5 ;
                postop_type = 5 ;
            }
    };

    struct blob_summary
    {
        std::size_t elf_header_size;
        std::size_t mv_header_size;
        std::size_t header_pad_size;
        std::size_t stage_section_size;
        std::size_t buffer_header_size;
        std::size_t buffer_data_size;
        std::size_t buffer_data_pad_size;
        std::size_t relocation_section_size;
        std::size_t weights_region_size;
        std::size_t bias_region_size;
        std::size_t weights_number_size;
        std::size_t tensor_number_size;
        std::size_t stage_count;
        std::size_t data_buffer_count;
        std::size_t elt_count;
        std::size_t input_size;
        std::size_t output_size;
        std::size_t blob_file_size;
        std::vector<std::size_t> relocbuf_list = {  } ;
        std::vector<std::size_t> relocadr_list = {  } ;
    };

    class Blob_buffer : public WBuffer
    {
        private:
            blob_summary blob_stats;

        public:
            RelocationTable reloc_table;
            Blob_buffer()
            {
                this->reloc_table = RelocationTable();
            }

            blob_summary getBlobSumm();


            // Calculate Blob Statistics
            std::size_t calc(mv::ControlModel& cm);

            void write_elf_header();

            void write_mv_header();

            void write_stage_section_header();

            void add_stage_IO_info(mv::Control::OpDFSIterator it, mv::Blob_stage conv_pool_stage);

            void write_stages(mv::ControlModel& cm);

            void write_buffer_section(mv::ControlModel& cm);

            void write_relocation_section(mv::ControlModel& cm);

            int get_blob_enum(mv::OpType o, bool NCE1=false);

    };   // end class blob_buffer

}

#endif // MV_BLOB_SERIALIZER_HPP_
