#ifndef MV_BLOB_SERIALIZER_HPP_
#define MV_BLOB_SERIALIZER_HPP_

/**
* serializer.hpp contains classes that output to file compute graph representations in various formats.
*
* @author Patrick Doyle, Ian Hunter
* @date 4/27/2018
*/
#include "include/mcm/op_model.hpp"

#include "include/mcm/utils/serializer/file_buffer.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/deployer/blob_serialization/myriadX_hardware_descriptors.hpp"
#include "include/mcm/deployer/blob_serialization/bTensor.hpp"
#include "include/mcm/target/target_descriptor.hpp"
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
            uint32_t next ;
            uint32_t op_type ;
            uint32_t implementation  ;
            uint32_t preop_type  ;
            uint32_t postop_type ;

            uint32_t radixX;
            uint32_t radixY;
            uint32_t radixStrideX;
            uint32_t radixStrideY;
            uint32_t padX;
            uint32_t padY;
            uint32_t padStyle;
            uint32_t dilation;

            uint32_t InputDimX;
            uint32_t InputDimY;
            uint32_t InputDimZ;
            uint32_t InputStrideX;
            uint32_t InputStrideY;
            uint32_t InputStrideZ;
            uint32_t InputOffset;
            uint32_t InputLocation;
            uint32_t InputDataType;
            uint32_t InputOrder;
            uint32_t Input1Offset;
            uint32_t Input1Location;
            uint32_t TBOffset;

            uint32_t OutputDimX;
            uint32_t OutputDimY;
            uint32_t OutputDimZ;
            uint32_t OutputStrideX;
            uint32_t OutputStrideY;
            uint32_t OutputStrideZ;
            uint32_t OutputOffset;
            uint32_t OutputLocation;
            uint32_t OutputDataType;
            uint32_t OutputOrder;

            uint32_t TapsDimX;
            uint32_t TapsDimY;
            uint32_t TapsDimZ;
            uint32_t TapsStrideX;
            uint32_t TapsStrideY;
            uint32_t TapsStrideZ;
            uint32_t TapsOffset;
            uint32_t TapsLocation;
            uint32_t TapsDataType;
            uint32_t TapsOrder;

            uint32_t BiasDimX;
            uint32_t BiasDimY;
            uint32_t BiasDimZ;
            uint32_t BiasStrideX;
            uint32_t BiasStrideY;
            uint32_t BiasStrideZ;
            uint32_t BiasOffset;
            uint32_t BiasLocation;
            uint32_t BiasDataType;
            uint32_t BiasOrder;

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
        uint32_t elf_header_size;
        uint32_t mv_header_size;
        uint32_t header_pad_size;
        uint32_t stage_section_size;
        uint32_t buffer_header_size;
        uint32_t buffer_data_size;
        uint32_t buffer_data_pad_size;
        uint32_t relocation_section_size;
        uint32_t weights_region_size;
        uint32_t bias_region_size;
        uint32_t weights_number_size;
        uint32_t tensor_number_size;
        uint32_t stage_count;
        uint32_t data_buffer_count;
        uint32_t elt_count;
        uint32_t input_size;
        uint32_t output_size;
        uint32_t blob_file_size;
        std::vector<std::uint32_t> relocbuf_list = {  } ;
        std::vector<std::uint32_t> relocadr_list = {  } ;
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
            uint32_t calc(mv::ControlModel& cm, mv::TargetDescriptor& td);

            void write_elf_header();

            void write_mv_header();

            void write_stage_section_header();

            void add_stage_IO_info(mv::Control::OpDFSIterator it, mv::Blob_stage conv_pool_stage);

            void write_stages(mv::ControlModel& cm);

            void write_ops(mv::ComputationModel& model, mv::TargetDescriptor& td);

            void write_buffer_section(mv::ControlModel& cm);

            void write_relocation_section(mv::ControlModel& cm);

    };   // end class blob_buffer

}

#endif // MV_BLOB_SERIALIZER_HPP_
