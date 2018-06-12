/**
* serializer.hpp contains classes that output to file compute graph representations in various formats.
*
* @author Patrick Doyle
* @date 4/27/2018
*/
#include "include/fathom/computation/model/op_model.hpp"
#include "include/fathom/deployer/mv_types.h"
#include "include/fathom/deployer/Fp16Convert.h"
#include "include/fathom/deployer/file_buffer.h"

namespace mv
{
/// List of supported graph serialization formats
enum serializer_mode
{
    mvblob_mode,
    json_mode,
    flatbuffers_mode,
    dot_mode
};

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
            padStyle = 1 ;
            dilation = 1 ;

            InputDimX = 32 ;
            InputDimY = 32 ;
            InputDimZ = 3 ;
            InputStrideX = 2 ;
            InputStrideY = 64 ;
            InputStrideZ = 2 ;
            InputOffset = 0 ;
            InputLocation = 1 ;
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
            TapsLocation = 3 ;
            TapsDataType = 0 ;
            TapsOrder = 3 ;

            BiasDimX = 0 ;
            BiasDimY = 0 ;
            BiasDimZ = 0 ;
            BiasStrideX = 0 ;
            BiasStrideY = 0 ;
            BiasStrideZ = 0 ;
            BiasOffset = 1 ;
            BiasLocation = 3 ;
            BiasDataType = 0 ;
            BiasOrder = 1 ;

            preop_type = 5 ;
            postop_type = 9 ;
        }
};

struct blob_summary {
    uint32_t elf_header_size;
    uint32_t mv_header_size;
    uint32_t header_pad_size;
    uint32_t stage_section_size;
    uint32_t buffer_header_size;
    uint32_t buffer_data_size;
    uint32_t relocation_section_size;
    uint32_t weights_region_size;
    uint32_t weights_region_pad_size;
    uint32_t bias_region_size;
    uint32_t params_region_size;
    uint32_t weights_number_size;
    uint32_t stage_count;
    uint32_t input_size;
    uint32_t output_size;
    uint32_t blob_file_size;
};


class Blob_buffer : public WBuffer
{
    private:
        blob_summary blob_stats;

    public:
        Blob_buffer()
        {
        }
        const mv::string conv_str = "conv" ;
        void calc(mv::ControlModel& cm)
        {
            // calculate blob statistics
            // set input size from compute model 
            blob_stats.input_size = cm.getFirst()->getOutput()->getShape().totalSize();

            // set fixed header sizes for blob
            blob_stats.elf_header_size = 34 ;
            blob_stats.mv_header_size = 40 ;
            uint32_t headers_data_size = blob_stats.elf_header_size+blob_stats.mv_header_size ;
            blob_stats.header_pad_size = align(headers_data_size,0x10)-headers_data_size;
            blob_stats.buffer_header_size = 0x10 ;
            blob_stats.weights_number_size = 2 ;          // TODO assume FP16 

            // parse compute model to determine stage dependent sizes
            // initialize values that will increase during parse of graph
            blob_stats.stage_count = 0 ;
            blob_stats.stage_section_size = 4*3 ;    // start count including 12 byte header 
            blob_stats.weights_region_size = 0 ;
            blob_stats.bias_region_size = 0 ;  
            blob_stats.params_region_size = 0 ; 

            for (mv::ControlContext::OpListIterator it = cm.getFirst(); it != cm.opEnd(); ++it)
            {
                if ( it->getOpType() == OpType::Conv2D )
                {
//                    std::cout << "calculating buffer sizes for convolution"<< std::endl;

                    // buffer data section for convolution has 3 regions: taps, bias, and params
                    // size of TAP region = align((roundUp(8,#kC)*kernelX*kernelY*kN)*dataSize),0x40)
                    //  TODO       BIAS region = align((#biases*dataSize),0x40) 
                    //  TODO       PARAMS region = align((#params*dataSize),0x40) 

                    // TAPS region
                    // calculate buffer sizes etc related to weights
                    uint32_t kernel_sizeX = it->getInput(1)->getShape()[0] ;
                    uint32_t kernel_sizeY = it->getInput(1)->getShape()[1] ;
                    uint32_t kernel_sizeC = it->getInput(1)->getShape()[2] ;
                    uint32_t kernel_sizeN = it->getInput(1)->getShape()[3] ;
                    uint32_t buffer_taps_weights_len = kernel_sizeX*kernel_sizeY*kernel_sizeC*kernel_sizeN;
                    uint32_t buffer_taps_weights_size = buffer_taps_weights_len*blob_stats.weights_number_size;
                    uint32_t weights_region_size = align(kernel_sizeC,8)*kernel_sizeX*kernel_sizeY*kernel_sizeN*blob_stats.weights_number_size ;
                    weights_region_size = align(weights_region_size,64) ;
//                    std::cout << "this weights_region_size = "<< weights_region_size << std::endl;
                    blob_stats.weights_region_size += weights_region_size ;
                    blob_stats.weights_region_pad_size = blob_stats.weights_region_size - buffer_taps_weights_size ;

                    blob_stats.stage_count++ ;
                    blob_stats.stage_section_size += (53*4) ;
                    blob_stats.bias_region_size += 64 ;                                
                    blob_stats.params_region_size += 64 ;    
                }
                
            }

            blob_stats.output_size = cm.getLast()->getInput(0)->getShape()[1] * cm.getLast()->getInput(0)->getShape()[2] ;
            blob_stats.stage_section_size = align(blob_stats.stage_section_size, 16) ;
            blob_stats.buffer_data_size = blob_stats.weights_region_size + blob_stats.bias_region_size + blob_stats.params_region_size ;

/*
            uint32_t number_of_convolutions = blob_stats.stage_count ;
            uint32_t relocation_section_header_size = 20 ;
            uint32_t blob_buffer_reloc_size = 16*number_of_convolutions ;
            uint32_t work_buffer_reloc_size = 0x10 * (blob_stats.stage_count-1);
            uint32_t relocation_section_size = work_buffer_reloc_size + blob_buffer_reloc_size + relocation_section_header_size ;
*/

            blob_stats.relocation_section_size = 20 + 16*blob_stats.stage_count + 16*(blob_stats.stage_count-1) ;
            blob_stats.blob_file_size = headers_data_size+blob_stats.header_pad_size+blob_stats.stage_section_size+blob_stats.buffer_header_size+blob_stats.buffer_data_size+blob_stats.relocation_section_size ;
        }

        void write_elf_header()
        {

            AddBytes(2, 0x0000);  // 0x00
            AddBytes(2, 0x0001);
            AddBytes(2, 0x0002);
            AddBytes(2, 0x0001);
            AddBytes(2, 0x0000);
            AddBytes(2, 0x0000);
            AddBytes(2, 0x0000);
            AddBytes(2, 0x0000);

            AddBytes(2, 0x0000);  // 0x10
            AddBytes(2, 0x0000);
            AddBytes(2, 0x0000);
            AddBytes(2, 0x0110);
            AddBytes(2, 0x0000);
            AddBytes(2, 0x0000);
            AddBytes(2, 0x0000);
            AddBytes(2, 0x0000);

            AddBytes(2, 0x0000);  // 0x20


/* disable zero elf header
            int j;
            const int elfhdr_length = 34 ;
            for (j=0; j< elfhdr_length; j++)
                {
                   AddBytes(1, 0x00);
                }
*/
/* temporarily disable valid elf header
            // E_IDENT
            AddBytes(4, 0x464c457f);// EI_MAG = x7f.ELF
            AddBytes(1, 0x01) ;     // EI_CLASS = 1
            AddBytes(1, 0x01) ;     // EI_DATA  = 1
            AddBytes(1, 0x01) ;     // EI_VERSION = 1
            AddBytes(10, 0x00) ;    // EI_OSABI, EI_ABIVERSION, EI_PAD = 0
            AddBytes(2, 0x0001) ;   // E_TYPE = 1
            AddBytes(2, 0x0002) ;   // E_MACHINE = 2
            AddBytes(1, 0x01);      // E_VERSION = 1
            AddBytes(15, 0x00) ;    // E_ENTRY, E_PHOFF, E_SHOFF, E_FLAGS = 0
            AddBytes(1, 0x30 ;      // E_EHSIZE = 48 (0x30)
            AddBytes(10, 0x00) ;    // pad
*/

        }

       void write_mv_header()
        {

            uint32_t mv_magic_number = 8708 ;
            uint32_t mv_version_major = 2 ;
            uint32_t mv_version_minor = 3 ;
            uint32_t mv_num_shaves = 1 ;

            uint32_t mv_stage_section_offset = blob_stats.elf_header_size+blob_stats.mv_header_size+blob_stats.header_pad_size ;

            uint32_t mv_buffer_section_offset = mv_stage_section_offset + blob_stats.stage_section_size ;
            uint32_t mv_relocation_offset = mv_buffer_section_offset + blob_stats.buffer_header_size + blob_stats.buffer_data_size ;
            uint32_t mv_permutation_enabled = 0x0000 ;
            
            AddBytes(4, mv_magic_number);

            AddBytes(4, blob_stats.blob_file_size);
            AddBytes(4, mv_version_major);
            AddBytes(4, mv_version_minor);
            AddBytes(4, mv_num_shaves);             // 0x32
            AddBytes(4, mv_stage_section_offset);
            AddBytes(4, mv_buffer_section_offset);
            AddBytes(4, mv_relocation_offset);
            AddBytes(4, blob_stats.input_size);
            AddBytes(4, mv_permutation_enabled);

            AddBytes(blob_stats.header_pad_size, 0x00);

        }

       void write_stage_section_header()
       {
            AddBytes(4, blob_stats.stage_count);   // 0x50
            AddBytes(4, blob_stats.stage_section_size);
            AddBytes(4, blob_stats.output_size);
       }

       void write_stages(mv::ControlModel& cm)
       {

            Blob_stage test_conv_stage ;

            uint32_t op_count = 0 ;
            for (mv::ControlContext::OpListIterator it = cm.getFirst(); it != cm.opEnd(); ++it)
            {
                if ( it->getOpType() == OpType::Conv2D )
                {
                    op_count++;
//                    std::cout << "writing stage for convolution"<< std::endl;

                    // this stage header
                    if (op_count == blob_stats.stage_count)
                    { 
                        test_conv_stage.next = 0x0 ;
                        test_conv_stage.OutputLocation = 0x00000002 ;
                    }
                    else
                    {
                        test_conv_stage.next = 0xe0 ;
                        test_conv_stage.OutputLocation = 0x00000004 ;
                    }

                    if (op_count == 1)
                    {
                        test_conv_stage.InputLocation = 0x00000001 ;
                    }
                    else
                    {
                        test_conv_stage.InputLocation = 0x00000004 ;
                        test_conv_stage.InputOffset++ ;
                    }

                    AddBytes(4, test_conv_stage.next);
                    
            AddBytes(4, test_conv_stage.op_type);    // 0x60
            AddBytes(4, test_conv_stage.implementation);

                    // operator specific info
                    AddBytes(4, it->getInput(1)->getShape()[0]); //radixX
                    AddBytes(4, it->getInput(1)->getShape()[1]); //radixY
                    AddBytes(4, it->getAttr("stride").getContent<mv::UnsignedVector2D>().e0); //strideX  (0x70)
                    AddBytes(4, it->getAttr("stride").getContent<mv::UnsignedVector2D>().e1); //strideY
                    // Ignore asymmetric padding (ignore elements elements p_r and p_b from padding = [p_l, p_r, p_t, p_b])
                    AddBytes(4, it->getAttr("padding").getContent<mv::UnsignedVector4D>().e0);  // padX
                    AddBytes(4, it->getAttr("padding").getContent<mv::UnsignedVector4D>().e2);  // padY
            AddBytes(4, test_conv_stage.padStyle);   // 0x80
            AddBytes(4, test_conv_stage.dilation);

                    // python helper push
                    int number_size = 2 ;  // TODO assume FP16
                    AddBytes(4, it->getInput(0)->getShape()[0]);  // input X-dimension size
                    AddBytes(4, it->getInput(0)->getShape()[1]);  // input Y-dimension size
                    AddBytes(4, it->getInput(0)->getShape()[2]);  // input Z-dimension size   (0x90)
                    AddBytes(4, number_size*it->getInput(0)->getShape()[2]);    // InputStrideX
                    AddBytes(4, number_size*it->getInput(0)->getShape()[2]*it->getInput(0)->getShape()[0]);  // InputStrideY
                    AddBytes(4, number_size*it->getInput(0)->getShape()[0]);   // InputStrideZ
            AddBytes(4, test_conv_stage.InputOffset);     //  0xa0
            AddBytes(4, test_conv_stage.InputLocation);
            AddBytes(4, test_conv_stage.InputDataType);
            AddBytes(4, test_conv_stage.InputOrder);

                    AddBytes(4, it->getOutput()->getShape()[1]);  // output X-dimension size  (0xb0)
                    AddBytes(4, it->getOutput()->getShape()[2]);  // output Y-dimension size
                    AddBytes(4, it->getOutput()->getShape()[0]);  // output Z-dimension size
            AddBytes(4, test_conv_stage.OutputStrideX);
                    AddBytes(4, number_size*it->getOutput()->getShape()[1]);   // 0xc0
            AddBytes(4, test_conv_stage.OutputStrideZ);
            AddBytes(4, test_conv_stage.OutputOffset);
            AddBytes(4, test_conv_stage.OutputLocation);
            AddBytes(4, test_conv_stage.OutputDataType);   //0xd0
            AddBytes(4, test_conv_stage.OutputOrder);

            AddBytes(4, it->getInput(1)->getShape()[0]*it->getInput(1)->getShape()[1]);
            AddBytes(4, it->getInput(1)->getShape()[3]);
            AddBytes(4, test_conv_stage.TapsDimZ);    // 0xe0
            AddBytes(4, number_size*it->getInput(1)->getShape()[3]);
            AddBytes(4, test_conv_stage.TapsStrideY);
            AddBytes(4, test_conv_stage.TapsStrideZ);
            AddBytes(4, test_conv_stage.TapsOffset);   // 0xf0
            AddBytes(4, test_conv_stage.TapsLocation);
            AddBytes(4, test_conv_stage.TapsDataType);
            AddBytes(4, test_conv_stage.TapsOrder);

            AddBytes(4, test_conv_stage.BiasDimX);   // 0x100
            AddBytes(4, test_conv_stage.BiasDimY);
            AddBytes(4, test_conv_stage.BiasDimZ);
            AddBytes(4, test_conv_stage.BiasStrideX);
            AddBytes(4, test_conv_stage.BiasStrideY);   // 0x110
            AddBytes(4, test_conv_stage.BiasStrideZ);
            AddBytes(4, test_conv_stage.BiasOffset);
            AddBytes(4, test_conv_stage.BiasLocation);
            AddBytes(4, test_conv_stage.BiasDataType);   // 0x120
            AddBytes(4, test_conv_stage.BiasOrder);

            AddBytes(4, test_conv_stage.preop_type);
            AddBytes(4, test_conv_stage.postop_type);    // 0x12c

                test_conv_stage.TapsOffset= test_conv_stage.TapsOffset+2 ;
                test_conv_stage.BiasOffset= test_conv_stage.BiasOffset+2 ;
                }
            }

            uint32_t total_stage_size = (0x50+(3*4)+(53*4*blob_stats.stage_count)) ;
            uint32_t buffer_section_offset = align(total_stage_size,0x10) ;
            uint32_t stage_pad_size = buffer_section_offset - total_stage_size ;

            AddBytes(stage_pad_size, 0x00);
        }

       void write_buffer_section(mv::ControlModel& cm)
       {
            uint32_t buffer_header_pad_size = 3 ;
            uint32_t buffer_header_pad_val = 0x002a ;
            uint8_t buffer_pad_val = 0x00 ;
            uint8_t buffer_wpad_val = 0x00 ;

            // buffer section header
            AddBytes(4, (blob_stats.buffer_header_size + blob_stats.buffer_data_size));

            for (unsigned i=0; i<buffer_header_pad_size; i++)
            {   
                AddBytes(4, buffer_header_pad_val);
            }

            for (mv::ControlContext::OpListIterator it = cm.getFirst(); it != cm.opEnd(); ++it)
            {
                if ( it->getOpType() == OpType::Conv2D )
                {
                    // buffer data section for convolution has 3 regions: taps, bias, and params
                    // size of TAP region = align((roundUp(8,#kernels)*kernelX*kernelY*kernelZ)*dataSize),0x40)
                    //  TODO       BIAS region = align((#biases*dataSize),0x40) 
                    //  TODO       PARAMS region = align((#params*dataSize),0x40) 

                    // TAPS region
                    // calculate buffer sizes etc related to weights
                    uint32_t kernel_sizeX = it->getInput(1)->getShape()[0] ; 
                    uint32_t kernel_sizeY = it->getInput(1)->getShape()[1] ; 
                    uint32_t kernel_sizeZ = it->getInput(1)->getShape()[2] ; 
                    uint32_t kernel_sizeN = it->getInput(1)->getShape()[3] ; 
                    uint32_t weights_number_size = 2 ;          // TODO assume FP16 
                    uint32_t buffer_taps_weights_len = kernel_sizeX*kernel_sizeY*kernel_sizeZ*kernel_sizeN; 
                    uint32_t buffer_taps_weights_size = buffer_taps_weights_len*blob_stats.weights_number_size;
                    uint32_t weights_region_size = align(kernel_sizeZ,8)*kernel_sizeX*kernel_sizeY*kernel_sizeN*blob_stats.weights_number_size ;
                    weights_region_size = align(weights_region_size,64) ;
                    uint32_t weights_region_pad_size = weights_region_size - buffer_taps_weights_size ;

                    // write weights and pad to file
                    for (unsigned i=0; i< buffer_taps_weights_len; i++)
                    {
                        uint16_t cur_weight = f32Tof16(it->getInput(1)->getData()[i]) ; 
                        AddBytes(weights_number_size, cur_weight) ; 
                    }

                    for (unsigned i=0; i< weights_region_pad_size; i++)
                    {
                        AddBytes(1, buffer_wpad_val);
                    }

                    // BIAS region
                    uint32_t bias_number_size = 2 ;             // TODO assume FP16 
                    uint16_t buffer_bias_val = f32Tof16(0.0f);  // TODO bias = 0 hardcoded 
                    uint32_t buffer_bias_values_len = 1;        // TODO use 1 for now (same bias all outputs)
                    uint32_t buffer_bias_values_size = buffer_bias_values_len*bias_number_size;   
                    uint32_t buffer_bias_region_size = align(buffer_bias_values_size,64) ;
                    uint32_t buffer_bias_pad_size = buffer_bias_region_size - (buffer_bias_values_size);
                    for (unsigned i=0; i< buffer_bias_values_len; i++)
                    {
                        AddBytes(bias_number_size, buffer_bias_val);
                    }
                    for (unsigned i=0; i< buffer_bias_pad_size; i++)
                    {
                        AddBytes(1, buffer_pad_val);
                    }

                    // PARAMS region
                    uint32_t params_number_size = 4 ;           // assume int32 for postop param 
                    uint32_t buffer_params_val = 0x00000001;    // TODO always use bias postop 
                    uint32_t buffer_params_values_len = 1;      // TODO use 1 for now (same bias all outputs)
                    uint32_t buffer_params_values_size = buffer_params_values_len*params_number_size; 
                    uint32_t buffer_params_region_size = align(buffer_params_values_size,64) ;
                    uint32_t buffer_params_pad_size = buffer_params_region_size - (buffer_params_values_size);
                    for (unsigned i=0; i< buffer_params_values_len; i++)
                    {
                        AddBytes(params_number_size, buffer_params_val);
                    }
                    for (unsigned i=0; i< buffer_params_pad_size; i++)
                    {
                        AddBytes(1, buffer_pad_val);
                    }
                }
            }
       }

       void write_relocation_section(mv::ControlModel& cm)
       {

            uint32_t number_of_convolutions = blob_stats.stage_count ;
            uint32_t relocation_section_header_size = 20 ;   
            uint32_t blob_buffer_reloc_size = 16*number_of_convolutions ;
            uint32_t work_buffer_reloc_size = 0x10 * (blob_stats.stage_count-1);
            uint32_t relocation_section_size = work_buffer_reloc_size + blob_buffer_reloc_size + relocation_section_header_size ;
            uint32_t blob_buffer_reloc_offset = blob_stats.blob_file_size - relocation_section_size + relocation_section_header_size ;
            uint32_t work_buffer_reloc_offset = blob_buffer_reloc_offset + blob_buffer_reloc_size ;

            // write relocation section header
            AddBytes(4, relocation_section_size );
            AddBytes(4, blob_buffer_reloc_offset);
            AddBytes(4, blob_buffer_reloc_size);
            AddBytes(4, work_buffer_reloc_offset);
            AddBytes(4, work_buffer_reloc_size);

            // write buffer data relocation info
            uint32_t running_offset = 0 ;
            for (mv::ControlContext::OpListIterator it = cm.getFirst(); it != cm.opEnd(); ++it)
            {
                if ( it->getOpType() == OpType::Conv2D )
                {
                    // calculate buffer sizes etc related to weights
                    uint32_t kernel_sizeX = it->getInput(1)->getShape()[0] ;
                    uint32_t kernel_sizeY = it->getInput(1)->getShape()[1] ;
                    uint32_t kernel_sizeC = it->getInput(1)->getShape()[2] ;
                    uint32_t kernel_sizeN = it->getInput(1)->getShape()[3] ;
                    uint32_t bias_region_size = 64 ;
                    uint32_t params_region_size = 64 ;
                    uint32_t weights_region_size = align(kernel_sizeC,8)*kernel_sizeX*kernel_sizeY*kernel_sizeN*blob_stats.weights_number_size ;
                    weights_region_size = align(weights_region_size,64) ;
                    // relocation section: blob buffer relocation information
                    // weights region 
                    AddBytes(4, running_offset);  // offset from start of buffer section
                    AddBytes(4, 0x00000003);          // memory type = heap/bss  
                    running_offset += weights_region_size ;
                    // bias region offset 
                    AddBytes(4, running_offset);
                    AddBytes(4, 0x00000003);          // memory type = heap/b
                    running_offset += bias_region_size + params_region_size ;
                }
            }

           for (unsigned j=0; j<(blob_stats.stage_count-1); j++)
           {
                // relocation section: work buffer relocation information
                AddBytes(4, 0x00000000);          // offset from start of work section
                AddBytes(4, 0x00000004);          // memory type =   
                //  
                AddBytes(4, 0x00000000);
                AddBytes(4, 0x00000004);               // memory type = heap/bss  
           }

        }

     };

/**
* @brief Serializer outputs verious representations of the compute graph. Initially moviduius binary blob format is supported.
*
* @param set_serialization_mode defines the output format of the graph
*/
    class Serializer 
    {

    private:
        serializer_mode output_format;
        Blob_buffer odata;

    public:
  
        Serializer(serializer_mode set_output_format) 
        {
            output_format = set_output_format;
        }

/**
* @brief serialize writes the specified format output file desecribing the compute model.
*
* @param graph_2_deploy (by reference) points to the graph you want to deploy
*/
        uint64_t serialize(mv::ControlModel& graph_2_deploy, const char* ofilename )
        {
            uint64_t fsize = 0 ;
            switch( output_format )
            {
                case mvblob_mode:
                    // 4 passes of graph: calculate, stages, buffer, reloc
                    // calculate sizes and offsets for headers
                    odata.calc(graph_2_deploy);
                    // write to file
                    odata.open(ofilename);
                    odata.write_elf_header();
                    odata.write_mv_header();
                    odata.write_stage_section_header();
                    odata.write_stages(graph_2_deploy); 
                    odata.write_buffer_section(graph_2_deploy);
                    odata.write_relocation_section(graph_2_deploy);
                    fsize = odata.End() ;
                break;
                default:
                    std::cout << "ERROR: unsupported deployment output format " << output_format << std::endl; 
                break;
            }
            return (fsize);
        }

        void print_mode()
        {
            std::cout << "serializer output mode= " << output_format << std::endl;
        }
};

}