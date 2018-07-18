/**
* serializer.hpp contains classes that output to file compute graph representations in various formats.
*
* @author Patrick Doyle
* @date 4/27/2018
*/
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/deployer/mv_types.h"
#include "include/mcm/deployer/Fp16Convert.h"
#include "include/mcm/deployer/file_buffer.h"
#include "include/mcm/pass/transform/fuse_relu.hpp"
#include "include/mcm/pass/transform/fuse_bias.hpp"
#include "include/mcm/pass/transform/fuse_scale.hpp"
#include "include/mcm/pass/transform/fuse_batch_norm.hpp"

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
//            padStyle = 1 ;
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
            TBOffset = 0 ;
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
            postop_type = 5 ;
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
    uint32_t tensor_number_size;
    uint32_t stage_count;
    uint32_t conv_count;
    uint32_t input_size;
    uint32_t output_size;
    uint32_t blob_file_size;
    std::vector<uint32_t> relocbuf_list = {  } ;
    std::vector<uint32_t> relocadr_list = {  } ;
};


class Blob_buffer : public WBuffer
{
    private:
        blob_summary blob_stats;

    public:
        Blob_buffer()
        {
        }
        void calc(mv::ControlModel& cm)
        {
            // calculate blob statistics
            // set input size from compute model
            blob_stats.input_size = cm.getFirst()->getOutputTensor(0)->getShape().totalSize();

            // set fixed header sizes for blob
            blob_stats.elf_header_size = 34 ;
            blob_stats.mv_header_size = 40 ;
            uint32_t headers_data_size = blob_stats.elf_header_size+blob_stats.mv_header_size ;
            blob_stats.header_pad_size = align(headers_data_size,0x10)-headers_data_size;
            blob_stats.buffer_header_size = 0x10 ;
            blob_stats.weights_number_size = 2 ;          // TODO assume FP16
            blob_stats.tensor_number_size = 2 ;          // TODO assume FP16

            // parse compute model to determine stage dependent sizes
            // initialize values that will increase during parse of graph
            blob_stats.stage_count = 0 ;
            blob_stats.conv_count = 0 ;
            blob_stats.stage_section_size = 4*3 ;    // start count including 12 byte header
            blob_stats.weights_region_size = 0 ;
            blob_stats.bias_region_size = 0 ;
            blob_stats.params_region_size = 0 ;

            for (mv::Control::OpDFSIterator it = cm.getFirst(); it != cm.opEnd(); ++it)
            {
                if (( it->getOpType() == OpType::Conv2D ) || ( it->getOpType() == OpType::FullyConnected ))
                {
                    uint32_t kernel_sizeX = 0 ;
                    uint32_t kernel_sizeY = 0 ;
                    uint32_t kernel_sizeZ = 0 ;
                    uint32_t kernel_sizeN = 0 ;

                    if ( it->getOpType() == OpType::FullyConnected )
                    {
//                        std::cout << "calculating buffer sizes for fully connected"<< std::endl;
                        kernel_sizeX = it->getInputTensor(1)->getShape().totalSize() ;
                        kernel_sizeY = 1 ;
                        kernel_sizeZ = 1 ;
                        kernel_sizeN = 1 ;
                        blob_stats.stage_section_size += (45*4) ;
                    }
                    else
                    {
//                        std::cout << "calculating buffer sizes for convolution"<< std::endl;
                        kernel_sizeX = it->getInputTensor(1)->getShape()[0] ;
                        kernel_sizeY = it->getInputTensor(1)->getShape()[1] ;
                        kernel_sizeZ = it->getInputTensor(1)->getShape()[2] ;
                        kernel_sizeN = it->getInputTensor(1)->getShape()[3] ;
                        blob_stats.stage_section_size += (53*4) ;
                    }

                    // buffer data section for convolution has 3 regions: taps, bias, and params
                    // size of TAP region = align((roundUp(8,#kC)*kernelX*kernelY*kN)*dataSize),0x40)
                    //  TODO       BIAS region = align((#biases*dataSize),0x40)
                    //  TODO       PARAMS region = align((#params*dataSize),0x40)

                    // TAPS region
                    // calculate buffer sizes etc related to weights
//                    std::cout << "this weights shape = " << kernel_sizeX << " " << kernel_sizeY << " " << kernel_sizeZ << " " << kernel_sizeN << std::endl;
                    uint32_t buffer_taps_weights_len = kernel_sizeX*kernel_sizeY*kernel_sizeZ*kernel_sizeN;
                    uint32_t buffer_taps_weights_size = buffer_taps_weights_len*blob_stats.weights_number_size;
                    uint32_t weights_region_size = align8(kernel_sizeN)*kernel_sizeX*kernel_sizeY*kernel_sizeZ*blob_stats.weights_number_size;
                    weights_region_size = align(weights_region_size,64) ;
//                    std::cout << "this weights_region_size = "<< weights_region_size << std::endl;
                    blob_stats.weights_region_size += weights_region_size ;
                    blob_stats.weights_region_pad_size = blob_stats.weights_region_size - buffer_taps_weights_size ;

                    // calculate buffer size related to bias
                    if (it->hasAttr("bias"))
                    {
                        uint32_t buffer_bias_values_len = it->getAttr("bias").getContent<mv::dynamic_vector<float>>().size() ;
                        buffer_bias_values_len = buffer_bias_values_len*blob_stats.weights_number_size;
                        blob_stats.bias_region_size += align(buffer_bias_values_len,64) ;
                    }
                    else
                    {
                        blob_stats.bias_region_size += 64 ;
                    }

                    blob_stats.stage_count++ ;
                    blob_stats.conv_count+=2 ;
                    blob_stats.params_region_size += 64 ;
                    if (it->hasAttr("postOpType"))
                    {
                        if (it->getAttr("postOpType").getContent<mv::OpType>() == mv::OpType::ReLU)
                        {
                            blob_stats.stage_section_size += (3*4) ;
                        }
                    }
                }
                else if (( it->getOpType() == OpType::MaxPool2D ) || ( it->getOpType() == OpType::AvgPool2D ))
                {
                    blob_stats.stage_count++ ;
                    blob_stats.stage_section_size += (3+7+20+2)*4 ;
                }
                else if (( it->getOpType() == OpType::Add) || ( it->getOpType() == OpType::Multiply))
                {
                    blob_stats.stage_count++ ;
                    blob_stats.stage_section_size += (3+32)*4 ;
                }
                else if (it->getOpType() == OpType::Softmax)
                {
                    blob_stats.stage_count++ ;
                    blob_stats.stage_section_size += (3+21+2)*4 ;
                }
                else if (it->getOpType() == OpType::ReLU)
                {
                    blob_stats.stage_count++ ;
                    blob_stats.stage_section_size += (3+3+20+2)*4 ;
                }
                else if (it->getOpType() == OpType::Scale)
                {
                    blob_stats.stage_count++ ;
                    blob_stats.stage_section_size += (3+32+10)*4 ;
                    blob_stats.conv_count++ ;   // uses buffer section (ala wts bias)
                    uint32_t buffer_bias_values_len = ( it->getInputTensor(1)->getShape().totalSize() ) *blob_stats.weights_number_size;
                    blob_stats.bias_region_size += align(buffer_bias_values_len,64) ;
                }

            }    // end traverse of graph

            blob_stats.output_size = cm.getLast()->getInputTensor(0)->getShape().totalSize();
//            std::cout << "output size = "<< blob_stats.output_size << std::endl;
            blob_stats.stage_section_size = align(blob_stats.stage_section_size, 16) ;
            blob_stats.buffer_data_size = blob_stats.weights_region_size + blob_stats.bias_region_size + blob_stats.params_region_size ;

            if (blob_stats.relocation_section_size % 16 == 0)
            {
                blob_stats.relocation_section_size += 16;
            }
            blob_stats.relocation_section_size = 20 + 8*blob_stats.conv_count + 16*(blob_stats.stage_count-1) ;
//            std::cout << "headers_data_size = "<< headers_data_size << std::endl;
//            std::cout << "blob_stats.header_pad_size = "<< blob_stats.header_pad_size << std::endl;
//            std::cout << "blob_stats.stage_section_size = "<< blob_stats.stage_section_size << std::endl;
//            std::cout << "blob_stats.buffer_header_size = "<< blob_stats.buffer_header_size << std::endl;
//            std::cout << "blob_stats.buffer_data_size = "<< blob_stats.buffer_data_size << std::endl;
//            std::cout << "    weights region = "<< blob_stats.weights_region_size << std::endl;
//            std::cout << "    bias region = "<< blob_stats.bias_region_size << std::endl;
//            std::cout << "    params region = "<< blob_stats.params_region_size << std::endl;
//            std::cout << "blob_stats.relocation_section_size = "<< blob_stats.relocation_section_size << std::endl;
            blob_stats.blob_file_size = headers_data_size+blob_stats.header_pad_size+blob_stats.stage_section_size+blob_stats.buffer_header_size+blob_stats.buffer_data_size+blob_stats.relocation_section_size ;
//            std::cout << "blob_stats.blob_file_size = "<< blob_stats.blob_file_size << std::endl;
//            std::cout << "finished calc" << std::endl;

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

       void add_stage_IO_info(mv::Control::OpDFSIterator it, mv::Blob_stage conv_pool_stage)
       {
           AddBytes(4, it->getInputTensor(0)->getShape()[0]);  // input X-dimension size
           AddBytes(4, it->getInputTensor(0)->getShape()[1]);  // input Y-dimension size
           AddBytes(4, it->getInputTensor(0)->getShape()[2]);  // input Z-dimension size   (0x90)
           AddBytes(4, blob_stats.tensor_number_size*it->getInputTensor(0)->getShape()[2]);    // InputStrideX
           AddBytes(4, blob_stats.tensor_number_size*it->getInputTensor(0)->getShape()[2]*it->getInputTensor(0)->getShape()[0]);  // InputStrideY
           AddBytes(4, blob_stats.tensor_number_size); // InputStrideZ
           AddBytes(4, conv_pool_stage.InputOffset);     //  0xa0
           AddBytes(4, conv_pool_stage.InputLocation);

//           std::cout << "added input offset, location : " << conv_pool_stage.InputOffset << " " << conv_pool_stage.InputLocation << std::endl;
           AddBytes(4, conv_pool_stage.InputDataType);
           AddBytes(4, conv_pool_stage.InputOrder);
           AddBytes(4, it->getOutputTensor(0)->getShape()[0]);  // output X-dimension size  (0xb0)
           AddBytes(4, it->getOutputTensor(0)->getShape()[1]);  // output Y-dimension size
           AddBytes(4, it->getOutputTensor(0)->getShape()[2]);  // output Z-dimension size
           AddBytes(4, blob_stats.tensor_number_size*it->getOutputTensor(0)->getShape()[2]);  // output stepX
           AddBytes(4, blob_stats.tensor_number_size*it->getOutputTensor(0)->getShape()[0]*it->getOutputTensor(0)->getShape()[2]);   // 0xc0
            AddBytes(4, conv_pool_stage.OutputStrideZ);
            AddBytes(4, conv_pool_stage.OutputOffset);
            AddBytes(4, conv_pool_stage.OutputLocation);
//           std::cout << "      output offset, location : " << conv_pool_stage.OutputOffset << " " << conv_pool_stage.OutputLocation << std::endl;
            AddBytes(4, conv_pool_stage.OutputDataType);   //0xd0
            AddBytes(4, conv_pool_stage.OutputOrder);

       }

       void write_stages(mv::ControlModel& cm)
       {
//            std::cout << "in write_stages" << std::endl;

            Blob_stage conv_pool_stage ;
            uint32_t op_count = 0 ;
            uint32_t next_offset = 12 ;
            uint32_t work_buffer_index = 4 ;
            std::vector<uint32_t> inbufnum_list = {  } ;
            std::vector<string> sourcename_list = {  } ;
            std::vector<uint32_t> outbufsiz_list = {  } ;
            std::vector<uint32_t> outbufadr_list = {  } ;
            std::vector<uint32_t> inbufadr_list = {  } ;
            std::vector<uint32_t> outbufnum_list = {  } ;
            std::vector<uint32_t> workbuffer_offsets = {  } ;
            mv::OpModel om(cm);

            // traverse graph to determine input buffer number, size and source node for each node in the computation
            // buffer numbers: 1=input 2=output 3=blob-buffersection 4+ = bss work buffer
            for (mv::Control::OpDFSIterator it = cm.getFirst(); it != cm.opEnd(); ++it)
            {

                std::cout << "processing " << mv::Printable::toString(it->getOpType()) << std::endl;
                if ((it->getOpType() == OpType::Conv2D)||(it->getOpType() == OpType::FullyConnected)||(it->getOpType() == OpType::AvgPool2D)||(it->getOpType() == OpType::MaxPool2D)||(it->getOpType() == OpType::Softmax)||(it->getOpType() == OpType::ReLU))
                {
                    // determine source
                    auto parentIt = om.getSourceOp(it->getInputTensor(0));

                    if (parentIt->getOpType() == OpType::Input)
                    {
                        inbufnum_list.push_back(1);
                        sourcename_list.push_back("Input");
                        std::cout << "pushing inbuffer_list 1 Input" << std::endl;
                    }
                    else
                    {
                        // determine if source buffer is already defined
                        bool branch_input = false ;
                        uint32_t source_list_size = sourcename_list.size() ; 
                        for ( uint32_t source_index = 0; source_index < source_list_size; source_index++ )
                        {
                            if (parentIt->getName() == sourcename_list[source_index])
                            {
                                branch_input = true ;
                                uint32_t common_node = inbufnum_list[source_index];
                                inbufnum_list.push_back(common_node);
                                sourcename_list.push_back(parentIt->getName());
                                std::cout << "pushing inbuffer_list (branch input) "<< work_buffer_index-1 << " " << parentIt->getName() << std::endl;
                            }
                        }
                        if (!branch_input)    // new buffer needed
                        {
                            inbufnum_list.push_back(work_buffer_index++);
                            sourcename_list.push_back(parentIt->getName());
                            std::cout << "pushing inbuffer_list "<< work_buffer_index-1 << " " << parentIt->getName() << std::endl;
                        }
                    }
                } // end single input operator case

                else if ((it->getOpType() == OpType::Add) || (it->getOpType() == OpType::Multiply) || (it->getOpType() == OpType::Scale))
                {

                    for ( int input_index = 0; input_index <2; input_index++ )
                    {
                        // determine source 0
                        auto parentIt = om.getSourceOp(it->getInputTensor(input_index));

                        if (parentIt->getOpType() == OpType::Input)
                        {
                            inbufnum_list.push_back(1);
                            sourcename_list.push_back("Input");
                            std::cout << "pushing inbuffer_list 1 Input 0" << std::endl;
                        }
                        else
                        {
                            inbufnum_list.push_back(work_buffer_index++);
                            sourcename_list.push_back(parentIt->getName());
                            std::cout << "pushing inbuffer_list "<< work_buffer_index-1 << " " << parentIt->getName() << std::endl;
                        }

                    }  // end for loop over inputs to ADD node
                }   // end 2-input, no pad  case

                else if (it->getOpType() == OpType::Output)
                {
                    // determine source
                    auto parentIt = om.getSourceOp(it->getInputTensor(0));

                    if (parentIt->getOpType() == OpType::Input)
                    {
                        inbufnum_list.push_back(1);
                        sourcename_list.push_back("Input");
                        std::cout << "pushing inbuffer_list 1 Input" << std::endl;
                    }
                    else
                    {
                        inbufnum_list.push_back(2);
                        sourcename_list.push_back(parentIt->getName());
                        std::cout << "pushing inbuffer_list 2 "<< parentIt->getName() << std::endl;
                    }
                } // end output node case

            }    // end input buffer calculation pass
//            std::cout << "    finished input buffer calculation pass" << std::endl;

            // traverse graph to determine output buffer number and size for each node in the computation
            // buffer numbers retreived from input buffer list with matching source name
            // store size and buffer number for later
            uint32_t running_offset = 0 ;
            for (mv::Control::OpDFSIterator it = cm.getFirst(); it != cm.opEnd(); ++it)
            {
                int work_buffer_size = 0 ;
                if (it->getOpType() != OpType::Output)
                {
                    int padX = 0;
                    int padY = 0;
//                    std::cout << "outbuf list traverse for  "<< it->getName() << std::endl;

                    if (it->getOpType() == OpType::Conv2D)
                    {
                        padX = ((((it->getInputTensor(1)->getShape()[0])/2)+1)*2) ;   // compatibility pad allowing conv output overrun
                        padY = 0;
//                        padX = it->getAttr("padding").getContent<mv::UnsignedVector4D>().e0 + 4 ;
//                        padY = it->getAttr("padding").getContent<mv::UnsignedVector4D>().e2 ;
//                        std::cout << "----kernel size/2 *2  "<< (((it->getInputTensor(1)->getShape()[0])/2)*2) << std::endl;
                    // determine size of work buffer including pad for alignment and number format size
                        int X_size = it->getOutputTensor(0)->getShape()[0]+padX ;
                        int Y_size = it->getOutputTensor(0)->getShape()[1]+padY ;
                        int C_size = it->getOutputTensor(0)->getShape()[2] ;
                        work_buffer_size=align(((X_size)*(Y_size)*(C_size)*blob_stats.tensor_number_size),64) ;
                    }
                    else if ((it->getOpType() == OpType::AvgPool2D)||(it->getOpType() == OpType::MaxPool2D))
                    {
                        padX = it->getAttr("padding").getContent<mv::UnsignedVector4D>().e0 + 2 ;
                        padY = it->getAttr("padding").getContent<mv::UnsignedVector4D>().e2 ;
//                        padX = (((it->getAttr("kSize").getContent<mv::UnsignedVector2D>().e0)/2)*1) ;
//                        padY = (((it->getAttr("kSize").getContent<mv::UnsignedVector2D>().e1)/2)*1) ;
                        // determine size of work buffer including pad for alignment and number format size
                        int X_size = it->getOutputTensor(0)->getShape()[0]+padX ;
                        int Y_size = it->getOutputTensor(0)->getShape()[1]+padY ;
                        int C_size = it->getOutputTensor(0)->getShape()[2] ;
                        work_buffer_size=align(((X_size)*(Y_size)*(C_size)*blob_stats.tensor_number_size),64) ;
                    } // end padded output operator case
                    else
                    {
                        work_buffer_size=align((it->getOutputTensor(0)->getShape().totalSize() * blob_stats.tensor_number_size),64) ;
                    }

                    // find output buffer name in source_name list
                    for ( uint32_t list_index = 0; list_index < inbufnum_list.size(); list_index++ )
                    {
                        if (sourcename_list[list_index] == it->getName())
                        {
                            if ((workbuffer_offsets.size() <= (inbufnum_list[list_index]-4))&&(inbufnum_list[list_index]>=4))
                            {
                                outbufnum_list.push_back(inbufnum_list[list_index]);
                                outbufsiz_list.push_back(work_buffer_size);
                                std::cout << "pushing outbuf list num size "<< inbufnum_list[list_index] << " " << work_buffer_size << std::endl;
                                std::cout << "   new  workbuffer_offsets[ "<< inbufnum_list[list_index]-4 << "]= " << running_offset << std::endl;
                                workbuffer_offsets.push_back(running_offset);
                                running_offset += work_buffer_size; 
                            }
                            else if (inbufnum_list[list_index]==2)
                            {
                                outbufnum_list.push_back(2);
                                outbufsiz_list.push_back(0);
                                std::cout << "pushing outbuf list num size 2 0"<< std::endl;
                           }
                        }
                    }   // end search inbuflist for match
                }   // end not-output case (no output tensor from output node)
            }   // end pass to fill outbuf lists

//            std::cout << "    finished output buffer calculation pass" << std::endl;

            // calculate address offset for each work buffer in inbufnum_list
            int buf_offset = 0 ;

            // find buffer size from outbufsiz_list
            for ( uint32_t inbuf_index = 0; inbuf_index < inbufnum_list.size(); inbuf_index++ )
            {
                uint32_t bufr2size = inbufnum_list[inbuf_index];
                if ( bufr2size >= 4 )
                {
                    inbufadr_list.push_back(workbuffer_offsets[bufr2size-4]);
                    std::cout << "pushing bufr adr(in) list: in_index bufnum off "<< inbuf_index << " " << bufr2size << " " << workbuffer_offsets[bufr2size-4] << std::endl;
                }     // end if WORK buffer
                else
                {
                    inbufadr_list.push_back(0);
                    std::cout << "pushing bufr adr(in) list: in_index bufnum off "<< inbuf_index << " " << bufr2size << " 0" << std::endl;   
                }
            }   // end inbuflist loop

            std::cout << "DEBUG: finished inbufadrlist " << std::endl;
            //  fill outbufadr_list
            for ( uint32_t obuf_index = 0; obuf_index < outbufnum_list.size(); obuf_index++ )
            {
                uint32_t bufr2copy = outbufnum_list[obuf_index];
                if (bufr2copy >= 4)
                {   
                    outbufadr_list.push_back(workbuffer_offsets[bufr2copy-4]);
                    std::cout << "pushing bufr adr(out) list: out_index bufnum off "<< obuf_index << " " << bufr2copy << " " << workbuffer_offsets[bufr2copy-4] << std::endl;   
                }     // end if WORK buffer
                else
                {
                    outbufadr_list.push_back(0);
                    std::cout << "pushing bufr adr(=out) list: out_index bufnum off "<< obuf_index << " " << bufr2copy << " 0" << std::endl;       
                }
            }   // end outbuf list loop

            // pass to output stage info -----------------------------------
            int outlist_index = 0 ;
            int inlist_index = 0 ;
            int reloc_index = 0 ;
            for (mv::Control::OpDFSIterator it = cm.getFirst(); it != cm.opEnd(); ++it)
            {

//                std::cout << "in write_stage_loop op_count = " << op_count << std::endl;

                if ( it->getOpType() == OpType::Conv2D )
                {

                    op_count++;
                    if (it->hasAttr("postOpType"))
                    {
                        if (it->getAttr("postOpType").getContent<mv::OpType>() == mv::OpType::ReLU)
                        {
                            next_offset += 0xd4 + (3*4) ;
                        }
                    }
                    else
                    {
                        next_offset += 0xd4 ;
                    }

                    // determine input and output buffer numbers. Save to blob_stats and write to stage section of blob
                    conv_pool_stage.InputLocation = inbufnum_list[inlist_index];
                    conv_pool_stage.OutputLocation = outbufnum_list[outlist_index];

                    // determine address offset to input buffer
                    if (conv_pool_stage.InputLocation != 1)
                    {
                        //  find input work buffer in output lists
                        for ( uint32_t olist_index = 0; olist_index < outbufnum_list.size(); olist_index++ )
                        {
                            if (conv_pool_stage.InputLocation == outbufnum_list[olist_index] )
                            {
                                blob_stats.relocbuf_list.push_back(outbufnum_list[olist_index]);
                                blob_stats.relocadr_list.push_back(outbufadr_list[olist_index]);
                            std::cout << "pushing reloc-table relindex bufnum siz "<< reloc_index << " " <<  outbufnum_list[olist_index] << " " << outbufsiz_list[olist_index] << std::endl;
                                conv_pool_stage.InputOffset = reloc_index++;
                            }
                        } // end search outbufnum list
                    }   // end node input is work buffer case
                    else
                    {
                       conv_pool_stage.InputOffset = 0 ;   // input to node is input to graph
                    }

                    // determine address offset to output buffer
                    if (conv_pool_stage.OutputLocation != 2)

                    {
                        blob_stats.relocbuf_list.push_back(outbufnum_list[outlist_index]); 
                        blob_stats.relocadr_list.push_back(outbufadr_list[outlist_index]); 
                            std::cout << "pushing reloc-table relindex bufnum siz "<< reloc_index << " " <<  outbufnum_list[outlist_index] << " " << outbufsiz_list[outlist_index] << std::endl;
                        conv_pool_stage.OutputOffset = reloc_index++;
                        conv_pool_stage.next = next_offset ;
                    }
                    else
                    {
                        conv_pool_stage.OutputOffset = 0 ;
                        conv_pool_stage.next = 0 ;
                    }

                    outlist_index++;
                    inlist_index++;

                    AddBytes(4, conv_pool_stage.next);
                    AddBytes(4, 0x00);                                // 0x60
                    AddBytes(4, conv_pool_stage.implementation);

                    // operator specific info
                    AddBytes(4, it->getInputTensor(1)->getShape()[0]); //radixX
                    AddBytes(4, it->getInputTensor(1)->getShape()[1]); //radixY
                    AddBytes(4, it->getAttr("stride").getContent<mv::UnsignedVector2D>().e0); //strideX  (0x70)
                    AddBytes(4, it->getAttr("stride").getContent<mv::UnsignedVector2D>().e1); //strideY
                    // Ignore asymmetric padding (ignore elements elements p_r and p_b from padding = [p_l, p_r, p_t, p_b])
                    AddBytes(4, it->getAttr("padding").getContent<mv::UnsignedVector4D>().e0);  // padX
                    AddBytes(4, it->getAttr("padding").getContent<mv::UnsignedVector4D>().e2);  // padY
                    AddBytes(4, conv_pool_stage.padStyle);   // 0x80
                    AddBytes(4, conv_pool_stage.dilation);

                    add_stage_IO_info(it, conv_pool_stage);

                    AddBytes(4, it->getInputTensor(1)->getShape()[0]*it->getInputTensor(1)->getShape()[1]);
                    AddBytes(4, it->getInputTensor(1)->getShape()[2]);
                    AddBytes(4, it->getInputTensor(1)->getShape()[3]);     // 0xe0   TapsDImZ

                    AddBytes(4, blob_stats.tensor_number_size*it->getInputTensor(1)->getShape()[2]*it->getInputTensor(1)->getShape()[3]);   // Taps step X
                    AddBytes(4, blob_stats.tensor_number_size*it->getInputTensor(1)->getShape()[3]);   // Taps step Y
                    AddBytes(4, conv_pool_stage.TapsStrideZ);
//                    AddBytes(4, conv_pool_stage.TapsOffset);   // 0xf0
                    AddBytes(4, conv_pool_stage.TBOffset);   // 0xf0
                    conv_pool_stage.TBOffset++ ; 
                    AddBytes(4, conv_pool_stage.TapsLocation);
                    AddBytes(4, conv_pool_stage.TapsDataType);
                    AddBytes(4, conv_pool_stage.TapsOrder);

                    AddBytes(4, conv_pool_stage.BiasDimX);   // 0x100
                    AddBytes(4, conv_pool_stage.BiasDimY);
                    AddBytes(4, conv_pool_stage.BiasDimZ);
                    AddBytes(4, conv_pool_stage.BiasStrideX);
                    AddBytes(4, conv_pool_stage.BiasStrideY);   // 0x110
                    AddBytes(4, conv_pool_stage.BiasStrideZ);
//                    AddBytes(4, conv_pool_stage.BiasOffset);
                    AddBytes(4, conv_pool_stage.TBOffset);
                    conv_pool_stage.TBOffset++ ;
                    AddBytes(4, conv_pool_stage.BiasLocation);
                    AddBytes(4, conv_pool_stage.BiasDataType);   // 0x120
                    AddBytes(4, conv_pool_stage.BiasOrder);

                    AddBytes(4, conv_pool_stage.preop_type);
                    std::cout << "debug 1" << std::endl;
                    if (it->hasAttr("postOpType"))
                    {
                    std::cout << "debug 2" << std::endl;
                        if (it->getAttr("postOpType").getContent<mv::OpType>() == mv::OpType::ReLU)
                        {
                    std::cout << "debug 3" << std::endl;
                            AddBytes(4, 0x06);    // 0x12c , postop relu
                            AddBytes(4, 0x00);
                            AddBytes(4, 0x00);
                            AddBytes(4, 0x00);
                        }
                    }
                    else
                    {
//                        AddBytes(4, 0x09);    // 0x12c , no postop
                        AddBytes(4, 0x05);    // 0x12c , no postop
                    }

//                    conv_pool_stage.TapsOffset= conv_pool_stage.TapsOffset+2 ;
//                    conv_pool_stage.BiasOffset= conv_pool_stage.BiasOffset+2 ;
                }   // end Conv case

                else if ( it->getOpType() == OpType::FullyConnected )
                {

//                    std::cout << "writing stage for FC" << std::endl;
                    op_count++;
                    if (it->hasAttr("postOpType"))
                    {
                        if (it->getAttr("postOpType").getContent<mv::OpType>() == mv::OpType::ReLU)
                        {
                            next_offset += 0xb4 + (3*4) ;
                        }
                    }
                    else
                    {
                        next_offset += 0xb4 ;
                    }

                    // determine input and output buffer numbers. Save to blob_stats and write to stage section of blob
                    conv_pool_stage.InputLocation = inbufnum_list[inlist_index];
                    conv_pool_stage.OutputLocation = outbufnum_list[outlist_index];

                    // determine address offset to input buffer
                    if (conv_pool_stage.InputLocation != 1)
                    {
                        //  find input work buffer in output lists
                        for ( uint32_t olist_index = 0; olist_index < outbufnum_list.size(); olist_index++ )
                        {
                            if (conv_pool_stage.InputLocation == outbufnum_list[olist_index] )
                            {
                                blob_stats.relocbuf_list.push_back(outbufnum_list[olist_index]);
                                blob_stats.relocadr_list.push_back(outbufadr_list[olist_index]);
                            std::cout << "pushing reloc-table relindex bufnum siz "<< reloc_index << " " <<  outbufnum_list[olist_index] << " " << outbufsiz_list[olist_index] << std::endl;
                                conv_pool_stage.InputOffset = reloc_index++;
                            }
                        } // end search outbufnum list 
                    }   // end node input is work buffer case 
                    else
                    {
                        conv_pool_stage.InputOffset = 0 ;   // input to node is input to graph
                    }

                    // determine address offset to output buffer
                    // determine address offset to output buffer
                    if (conv_pool_stage.OutputLocation != 2)
                    {
                        blob_stats.relocbuf_list.push_back(outbufnum_list[outlist_index]);
                        blob_stats.relocadr_list.push_back(outbufadr_list[outlist_index]);
                            std::cout << "pushing reloc-table relindex bufnum siz "<< reloc_index << " " <<  outbufnum_list[outlist_index] << " " << outbufsiz_list[outlist_index] << std::endl;
                        conv_pool_stage.OutputOffset = reloc_index++;
                        conv_pool_stage.next = next_offset ;
                    }
                    else
                    {
                        conv_pool_stage.OutputOffset = 0 ;
                        conv_pool_stage.next = 0 ;
                    }

                    outlist_index++;
                    inlist_index++;

                    AddBytes(4, conv_pool_stage.next);
                    AddBytes(4, 0x04);                                // 0x60  opcode for FC
                    AddBytes(4, conv_pool_stage.implementation);

//                    std::cout << "writing stage IO info for FC" << std::endl;

                    AddBytes(4, it->getInputTensor(0)->getShape()[0]);  // input X-dimension size
                    AddBytes(4, it->getInputTensor(0)->getShape()[1]);  // input Y-dimension size
                    AddBytes(4, it->getInputTensor(0)->getShape()[2]);  // input Z-dimension size   (0x90)
                    AddBytes(4, blob_stats.tensor_number_size*it->getInputTensor(0)->getShape()[2]);    // InputStrideX
                    AddBytes(4, blob_stats.tensor_number_size*it->getInputTensor(0)->getShape()[2]*it->getInputTensor(0)->getShape()[0]);  // InputStrideY
                    AddBytes(4, blob_stats.tensor_number_size); // InputStrideZ
                    AddBytes(4, conv_pool_stage.InputOffset);     //  0xa0
                    AddBytes(4, conv_pool_stage.InputLocation);

//                    std::cout << "added input offset, location : " << conv_pool_stage.InputOffset << " " << conv_pool_stage.InputLocation << std::endl;
                    AddBytes(4, conv_pool_stage.InputDataType);
                    AddBytes(4, conv_pool_stage.InputOrder);

                    AddBytes(4, it->getOutputTensor(0)->getShape().totalSize());  // output X-dimension size  (0xb0)
                    AddBytes(4, 0x01);  // output Y-dimension size
                    AddBytes(4, 0x01);  // output Z-dimension size
                    AddBytes(4, blob_stats.tensor_number_size);  // output stepX 
                    AddBytes(4, blob_stats.tensor_number_size*it->getOutputTensor(0)->getShape().totalSize());   // 0xc0
                    AddBytes(4, conv_pool_stage.OutputStrideZ);
                    AddBytes(4, conv_pool_stage.OutputOffset);
                    AddBytes(4, conv_pool_stage.OutputLocation);
//                    std::cout << "      output offset, location : " << conv_pool_stage.OutputOffset << " " << conv_pool_stage.OutputLocation << std::endl;
                    AddBytes(4, conv_pool_stage.OutputDataType);   //0xd0
                    AddBytes(4, conv_pool_stage.OutputOrder);

                    AddBytes(4, it->getInputTensor(1)->getShape().totalSize());   // TAPS dim X
                    AddBytes(4, 0x01);
                    AddBytes(4, 0x01 );     // 0xe0   TapsDImZ

                    AddBytes(4, blob_stats.tensor_number_size*it->getInputTensor(1)->getShape().totalSize());   // Taps step X
                    AddBytes(4, blob_stats.tensor_number_size);   // Taps step Y
                    AddBytes(4, conv_pool_stage.TapsStrideZ);
//                    AddBytes(4, conv_pool_stage.TapsOffset);   // 0xf0
                    AddBytes(4, conv_pool_stage.TBOffset);   // 0xf0
                    conv_pool_stage.TBOffset++ ;
                    AddBytes(4, conv_pool_stage.TapsLocation);
                    AddBytes(4, conv_pool_stage.TapsDataType);
                    AddBytes(4, conv_pool_stage.TapsOrder);

                    AddBytes(4, conv_pool_stage.BiasDimX);   // 0x100
                    AddBytes(4, conv_pool_stage.BiasDimY);
                    AddBytes(4, conv_pool_stage.BiasDimZ);
                    AddBytes(4, conv_pool_stage.BiasStrideX);
                    AddBytes(4, conv_pool_stage.BiasStrideY);   // 0x110
                    AddBytes(4, conv_pool_stage.BiasStrideZ);
//                    AddBytes(4, conv_pool_stage.BiasOffset);
                    AddBytes(4, conv_pool_stage.TBOffset);
                    conv_pool_stage.TBOffset++ ;
                    AddBytes(4, conv_pool_stage.BiasLocation);
                    AddBytes(4, conv_pool_stage.BiasDataType);   // 0x120
                    AddBytes(4, conv_pool_stage.BiasOrder);

                    AddBytes(4, conv_pool_stage.preop_type);
                    if (it->hasAttr("postOpType"))
                    {
                        if (it->getAttr("postOpType").getContent<mv::OpType>() == mv::OpType::ReLU)
                        {
                            AddBytes(4, 0x06);    // 0x12c , postop relu
                            AddBytes(4, 0x00);
                            AddBytes(4, 0x00);
                            AddBytes(4, 0x00);
                        }
                    }
                    else
                    {
//                        AddBytes(4, 0x09);    // 0x12c , no postop
                        AddBytes(4, 0x05);    // 0x12c , no postop
                    }

//                    conv_pool_stage.TapsOffset= conv_pool_stage.TapsOffset+2 ;
//                    conv_pool_stage.BiasOffset= conv_pool_stage.BiasOffset+2 ;

//                    std::cout << "finished writing stage for FC" << std::endl;
                }   // end fully connected case

                else if ( it->getOpType() == OpType::Softmax )
                {

                    op_count++;
                    next_offset += 0x68 ;

                    // determine input and output buffer numbers. Save to blob_stats and write to stage section of blob
                    conv_pool_stage.InputLocation = inbufnum_list[inlist_index];
                    conv_pool_stage.OutputLocation = outbufnum_list[outlist_index];

                    // determine address offset to input buffer
                    if (conv_pool_stage.InputLocation != 1)
                    {
                        //  find input work buffer in output lists
                        for ( uint32_t olist_index = 0; olist_index < outbufnum_list.size(); olist_index++ )
                        {
                            if (conv_pool_stage.InputLocation == outbufnum_list[olist_index] )
                            {
                                blob_stats.relocbuf_list.push_back(outbufnum_list[olist_index]);
                                blob_stats.relocadr_list.push_back(outbufadr_list[olist_index]);
                            std::cout << "pushing reloc-table relindex bufnum siz "<< reloc_index << " " <<  outbufnum_list[olist_index] << " " << outbufsiz_list[olist_index] << std::endl;
                                conv_pool_stage.InputOffset = reloc_index++;
                            }
                        } // end search outbufnum list
                    }   // end node input is work buffer case
                    else
                    {
                        conv_pool_stage.InputOffset = 0 ;   // input to node is input to graph
                    }

                    // determine address offset to output buffer
                    if (conv_pool_stage.OutputLocation != 2)
                    {
                        blob_stats.relocbuf_list.push_back(outbufnum_list[outlist_index]);
                        blob_stats.relocadr_list.push_back(outbufadr_list[outlist_index]);
                            std::cout << "pushing reloc-table relindex bufnum siz "<< reloc_index << " " <<  outbufnum_list[outlist_index] << " " << outbufsiz_list[outlist_index] << std::endl;
                        conv_pool_stage.OutputOffset = reloc_index++;
                        conv_pool_stage.next = next_offset ;
                    }
                    else
                    {
                        conv_pool_stage.OutputOffset = 0 ;
                        conv_pool_stage.next = 0 ;
                    }

                    outlist_index++;
                    inlist_index++;

                    AddBytes(4, conv_pool_stage.next);
                    AddBytes(4, 0x03);   // opcode for softmax
                    AddBytes(4, conv_pool_stage.implementation);

                    // operator specific info
                    AddBytes(4, 0x01); // softmax axis

//                    std::cout << "writing IO for softmax" << std::endl;
                    AddBytes(4, it->getInputTensor(0)->getShape().totalSize());  // input X-dimension size
                    AddBytes(4, 1);  // input Y-dimension size
                    AddBytes(4, 1);  // input Z-dimension size   (0x90)
                    AddBytes(4, blob_stats.tensor_number_size);    // InputStrideX
                    AddBytes(4, blob_stats.tensor_number_size*it->getInputTensor(0)->getShape().totalSize());  // InputStrideY
                    AddBytes(4, blob_stats.tensor_number_size); // InputStrideZ
                    AddBytes(4, conv_pool_stage.InputOffset);     //  0xa0
                    AddBytes(4, conv_pool_stage.InputLocation);

//                    std::cout << "added input offset, location : " << conv_pool_stage.InputOffset << " " << conv_pool_stage.InputLocation << std::endl;
                    AddBytes(4, conv_pool_stage.InputDataType);
                    AddBytes(4, conv_pool_stage.InputOrder);
                    AddBytes(4, it->getOutputTensor(0)->getShape().totalSize());  // output X-dimension size  (0xb0)
                    AddBytes(4, 1);  // output Y-dimension size
                    AddBytes(4, 1);  // output Z-dimension size
                    AddBytes(4, blob_stats.tensor_number_size);  // output stepX 
                    AddBytes(4, blob_stats.tensor_number_size*it->getOutputTensor(0)->getShape().totalSize());   // 0xc0
                    AddBytes(4, conv_pool_stage.OutputStrideZ);
                    AddBytes(4, conv_pool_stage.OutputOffset);
                    AddBytes(4, conv_pool_stage.OutputLocation);
//                    std::cout << "      output offset, location : " << conv_pool_stage.OutputOffset << " " << conv_pool_stage.OutputLocation << std::endl;
                    AddBytes(4, conv_pool_stage.OutputDataType);   //0xd0
                    AddBytes(4, conv_pool_stage.OutputOrder);

//                    std::cout << "wrote stage IO info for softmax" << std::endl;

                    AddBytes(4, conv_pool_stage.preop_type);
                    AddBytes(4, conv_pool_stage.postop_type);

                }    // end softmax case
                else if ( it->getOpType() == OpType::ReLU )
                {

//                    std::cout << "writing stage for ReLU" << std::endl;
                    op_count++;
                    next_offset += 0x70 ;

                    // determine input and output buffer numbers. Save to blob_stats and write to stage section of blob
                    conv_pool_stage.InputLocation = inbufnum_list[inlist_index];
                    conv_pool_stage.OutputLocation = outbufnum_list[outlist_index];

                    // determine address offset to input buffer
                    if (conv_pool_stage.InputLocation != 1)
                    {
                        //  find input work buffer in output lists
                        for ( uint32_t olist_index = 0; olist_index < outbufnum_list.size(); olist_index++ )
                        {
                            if (conv_pool_stage.InputLocation == outbufnum_list[olist_index] )
                            {
                                blob_stats.relocbuf_list.push_back(outbufnum_list[olist_index]);
                                blob_stats.relocadr_list.push_back(outbufadr_list[olist_index]);
                            std::cout << "pushing reloc-table relindex bufnum siz "<< reloc_index << " " <<  outbufnum_list[olist_index] << " " << outbufsiz_list[olist_index] << std::endl;
                                conv_pool_stage.InputOffset = reloc_index++;
                            }
                        } // end search outbufnum list
                    }   // end node input is work buffer case
                    else
                    {
                        conv_pool_stage.InputOffset = 0 ;   // input to node is input to graph
                    }

                    // determine address offset to output buffer
                    if (conv_pool_stage.OutputLocation != 2)
                    {
                        blob_stats.relocbuf_list.push_back(outbufnum_list[outlist_index]);
                        blob_stats.relocadr_list.push_back(outbufadr_list[outlist_index]);
                            std::cout << "pushing reloc-table relindex bufnum siz "<< reloc_index << " " <<  outbufnum_list[outlist_index] << " " << outbufsiz_list[outlist_index] << std::endl;
//                            std::cout << "conv_pool_stage.OutputLocation= "<< conv_pool_stage.OutputLocation << std::endl;
                        conv_pool_stage.OutputOffset = reloc_index++;
                        conv_pool_stage.next = next_offset ;
                    }
                    else
                    {
                        conv_pool_stage.OutputOffset = 0 ;
                        conv_pool_stage.next = 0 ;
                    }

                    outlist_index++;
                    inlist_index++;

                    AddBytes(4, conv_pool_stage.next);
                    AddBytes(4, 0x06);   // opcode for ReLU
                    AddBytes(4, conv_pool_stage.implementation);

                    // operator specific info
                    AddBytes(4, 0x00); // OpX

//                    std::cout << "writing IO for relu" << std::endl;
                    add_stage_IO_info(it, conv_pool_stage);
                    AddBytes(4, 0x00); // post stride x
                    AddBytes(4, 0x00); // post stride y

                    AddBytes(4, conv_pool_stage.preop_type);
                    AddBytes(4, conv_pool_stage.postop_type);
                }    // end relu case
                else if ( it->getOpType() == OpType::MaxPool2D )
                {
                    op_count++;
                    next_offset += 0x80 ;

                    // determine input and output buffer numbers. Save to blob_stats and write to stage section of blob
                    conv_pool_stage.InputLocation = inbufnum_list[inlist_index];
                    conv_pool_stage.OutputLocation = outbufnum_list[outlist_index];

                    // determine address offset to input buffer
                    if (conv_pool_stage.InputLocation != 1)
                    {
                        //  find input work buffer in output lists
                        for ( uint32_t olist_index = 0; olist_index < outbufnum_list.size(); olist_index++ )
                        {
                            if (conv_pool_stage.InputLocation == outbufnum_list[olist_index] )
                            {
                                blob_stats.relocbuf_list.push_back(outbufnum_list[olist_index]);
                                blob_stats.relocadr_list.push_back(outbufadr_list[olist_index]);
                                std::cout << "pushing reloc-table MPin "<< reloc_index << " " << outbufnum_list[olist_index] << " " << outbufsiz_list[olist_index] << std::endl;
                                conv_pool_stage.InputOffset = reloc_index++;
                                break;
                            }
                        } // end search outbufnum list
                    }   // end node input is work buffer case
                    else
                    {
                        conv_pool_stage.InputOffset = 0 ;   // input to node is input to graph
                    }
                    // determine address offset to output buffer
                    if (conv_pool_stage.OutputLocation != 2)
                    {
                        blob_stats.relocbuf_list.push_back(outbufnum_list[outlist_index]);
                        blob_stats.relocadr_list.push_back(outbufadr_list[outlist_index]);
                    std::cout << "pushing reloc-table MPout "<< reloc_index << " " << outbufnum_list[outlist_index] << " " << outbufsiz_list[outlist_index] << std::endl;
                    std::cout << "   outputlocation outlist_index "<< conv_pool_stage.OutputLocation << ' ' << outlist_index << std::endl;
                        conv_pool_stage.OutputOffset = reloc_index++;
                        conv_pool_stage.next = next_offset ;
                    }
                    else
                    {
                        conv_pool_stage.OutputOffset = 0 ;
                        conv_pool_stage.next = 0 ;
                    }

                    outlist_index++;
                    inlist_index++;

                    AddBytes(4, conv_pool_stage.next);
                    AddBytes(4, 1);             // opcode for maxpool is 1
//                    std::cout << "writing opcode=1 for MP " << std::endl;
                    AddBytes(4, conv_pool_stage.implementation);

                    // operator specific info
                    AddBytes(4, it->getAttr("kSize").getContent<mv::UnsignedVector2D>().e0); // radix X
                    AddBytes(4, it->getAttr("kSize").getContent<mv::UnsignedVector2D>().e1); // radix Y (0x140)
                    AddBytes(4, it->getAttr("stride").getContent<mv::UnsignedVector2D>().e0); //strideX
                    AddBytes(4, it->getAttr("stride").getContent<mv::UnsignedVector2D>().e1); //strideY
// TODO temp TF pad                    AddBytes(4, it->getAttr("padding").getContent<mv::UnsignedVector4D>().e0);  // padX
// TODO temp TF pad                    AddBytes(4, it->getAttr("padding").getContent<mv::UnsignedVector4D>().e2);  // padY
                    AddBytes(4, 0x00);   // padX
                    AddBytes(4, 0x00);   // padY 0x150
                    AddBytes(4, 0x02);   // padstyle

                    add_stage_IO_info(it, conv_pool_stage);
                    AddBytes(4, conv_pool_stage.preop_type);
                    AddBytes(4, 0x05);    // 0x1ac  postop type

                }
                else if ( it->getOpType() == OpType::AvgPool2D )
                {
                    op_count++;
                    next_offset += 0x80 ;

                    // determine input and output buffer numbers. Save to blob_stats and write to stage section of blob
                    conv_pool_stage.InputLocation = inbufnum_list[inlist_index];
                    conv_pool_stage.OutputLocation = outbufnum_list[outlist_index];

                    // determine address offset to input buffer
                    if (conv_pool_stage.InputLocation != 1)
                    {
                        //  find input work buffer in output lists
                        for ( uint32_t olist_index = 0; olist_index < outbufnum_list.size(); olist_index++ )
                        {
                            if (conv_pool_stage.InputLocation == outbufnum_list[olist_index] )
                            {
                                blob_stats.relocbuf_list.push_back(outbufnum_list[olist_index]);
                                blob_stats.relocadr_list.push_back(outbufadr_list[olist_index]);
                    std::cout << "pushing reloc-table "<< reloc_index << outbufnum_list[olist_index] << " " << outbufsiz_list[olist_index] << std::endl;
                        conv_pool_stage.InputOffset = reloc_index++;
                            }
                        } // end search outbufnum list
                    }   // end node input is work buffer case
                    else
                    {
                        conv_pool_stage.InputOffset = 0 ;   // input to node is input to graph
                    }

                    // determine address offset to output buffer
                    if (conv_pool_stage.OutputLocation != 2)
                    {
                        blob_stats.relocbuf_list.push_back(outbufnum_list[outlist_index]);
                        blob_stats.relocadr_list.push_back(outbufadr_list[outlist_index]);
                    std::cout << "pushing reloc-table "<< reloc_index << " "  << outbufnum_list[outlist_index] << " " << outbufsiz_list[outlist_index] << std::endl;
                        conv_pool_stage.OutputOffset = reloc_index++;
                        conv_pool_stage.next = next_offset ;
                    }
                    else
                    {
                        conv_pool_stage.OutputOffset = 0 ;
                        conv_pool_stage.next = 0 ;
                    }

                    outlist_index++;
                    inlist_index++;

                    AddBytes(4, conv_pool_stage.next);
                    AddBytes(4, 0x02);     // operation type for avgpool
//                    std::cout << "writing opcode=2 for AP " << std::endl;
                    AddBytes(4, conv_pool_stage.implementation);

                    // operator specific info
                    AddBytes(4, it->getAttr("kSize").getContent<mv::UnsignedVector2D>().e0); // radix X
                    AddBytes(4, it->getAttr("kSize").getContent<mv::UnsignedVector2D>().e1); // radix Y (0x140)
                    AddBytes(4, it->getAttr("stride").getContent<mv::UnsignedVector2D>().e0); //strideX
                    AddBytes(4, it->getAttr("stride").getContent<mv::UnsignedVector2D>().e1); //strideY
// TODO temp TF pad                    AddBytes(4, it->getAttr("padding").getContent<mv::UnsignedVector4D>().e0);  // padX
// TODO temp TF pad                    AddBytes(4, it->getAttr("padding").getContent<mv::UnsignedVector4D>().e2);  // padY
                    AddBytes(4, 0x00);   // padX
                    AddBytes(4, 0x00);   // padY 0x150
                    AddBytes(4, 0x03);   // padstyle

                    add_stage_IO_info(it, conv_pool_stage);
                    AddBytes(4, conv_pool_stage.preop_type);
                    AddBytes(4, 0x05);    // 0x1ac  postop type
                }
                else if (( it->getOpType() == OpType::Add ) || ( it->getOpType() == OpType::Multiply ) || ( it->getOpType() == OpType::Scale ))
                {
                    op_count++;
                    next_offset += 0x8c ;

                    // determine input and output buffer numbers. Save to blob_stats and write to stage section of blob
                    conv_pool_stage.OutputLocation = outbufnum_list[outlist_index];
                    uint32_t this_inputLocation ;
                    uint32_t this_inputOffset ;

                    //  write reloc table entry for 2 inputs
                    for ( int input_index = 0; input_index < 2; input_index++ )
                    {
                        if (( it->getOpType() == OpType::Scale )&&(input_index==1))
                        {
                            this_inputLocation = 3;  // second input to scale is located in the blob buff (wts-bias) 
                        }
                        else
                        {
                            this_inputLocation = inbufnum_list[inlist_index+input_index];   // input located in work buffer or input
                        }
                        // determine address offset to input buffer
                        if (this_inputLocation >= 4)
                        {
                            //  find input work buffer in output lists
                            for ( uint32_t olist_index = 0; olist_index < outbufnum_list.size(); olist_index++ )
                            {
                                if (this_inputLocation == outbufnum_list[olist_index] )
                                {
                                    blob_stats.relocbuf_list.push_back(outbufnum_list[olist_index]);
                                    blob_stats.relocadr_list.push_back(outbufadr_list[olist_index]);
                                    std::cout << "pushing reloc-table (add) "<< reloc_index << " " << outbufnum_list[olist_index] << " " << outbufsiz_list[olist_index] << std::endl;
                                    std::cout << "        olistIndex outbufnum_list.size "<< olist_index << " " << outbufnum_list.size() << std::endl;
                                    this_inputOffset = reloc_index++;
                                }
                            } // end search outbufnum list
                        }   // end node input is work buffer case
                        else
                        {
                            this_inputOffset = 0 ;   // input to node is input to graph
                        }

                        // 2nd input stage info is written as a TapsBuffer
                        if (input_index == 0)
                        {
                            conv_pool_stage.InputLocation = this_inputLocation;
                            conv_pool_stage.InputOffset = this_inputOffset;
                        }
                        else
                        {
                            conv_pool_stage.Input1Location = this_inputLocation;
                            conv_pool_stage.Input1Offset = this_inputOffset;
                        }

                    }   // end 2 input loop

                    // determine address offset to output buffer
                    if (conv_pool_stage.OutputLocation != 2)
                    {
                        blob_stats.relocbuf_list.push_back(outbufnum_list[outlist_index]);
                        blob_stats.relocadr_list.push_back(outbufadr_list[outlist_index]);
                    std::cout << "pushing reloc-table "<< reloc_index << " " << outbufnum_list[outlist_index] << " " << outbufsiz_list[outlist_index] << std::endl;
                        conv_pool_stage.OutputOffset = reloc_index++;
                        conv_pool_stage.next = next_offset ;
                    }
                    else
                    {
                        conv_pool_stage.OutputOffset = 0 ;
                        conv_pool_stage.next = 0 ;
                    }

                    outlist_index++;
                    inlist_index++;
                    inlist_index++;

                    AddBytes(4, conv_pool_stage.next);

                    if (it->getOpType() == OpType::Add)
                    {
                        AddBytes(4, 0x0c);     // operation type element-wise Add
                    }
                    else if (it->getOpType() == OpType::Multiply)
                    {
                        AddBytes(4, 0x0d);     // operation type element-wise Multiply
                    }
                    else
                    {
                        AddBytes(4, 0x0f);     // operation type vector Scale
                        next_offset += 0x28 ;
                    }

                    AddBytes(4, conv_pool_stage.implementation);

                    // operator specific info
                    add_stage_IO_info(it, conv_pool_stage);

                    if (it->getOpType() == OpType::Scale)
                    {
                        // 2nd input info 
//                        AddBytes(4, it->getInputTensor(1)->getShape().totalSize());  // input X-dimension size
                        AddBytes(4, 0x00);  // input X-dimension size
                        AddBytes(4, 1);  // input Y-dimension size
                        AddBytes(4, 1);  // input Z-dimension size   (0x90)
                        AddBytes(4, blob_stats.tensor_number_size);    // InputStrideX
                        AddBytes(4, blob_stats.tensor_number_size*it->getInputTensor(1)->getShape().totalSize());  // InputStrideY
                        AddBytes(4, blob_stats.tensor_number_size); // InputStrideZ

                        AddBytes(4, conv_pool_stage.TBOffset);      // 2nd input
                        conv_pool_stage.TBOffset++; 
                        AddBytes(4, 3);    // 2nd location is bias region of blob buffer
                        AddBytes(4, conv_pool_stage.OutputDataType);
                        AddBytes(4, 3);   // output order

                        AddBytes(4, conv_pool_stage.BiasDimX);   // 0x100
                        AddBytes(4, conv_pool_stage.BiasDimY);
                        AddBytes(4, conv_pool_stage.BiasDimZ);
                        AddBytes(4, conv_pool_stage.BiasStrideX);
                        AddBytes(4, conv_pool_stage.BiasStrideY);   // 0x110
                        AddBytes(4, conv_pool_stage.BiasStrideZ);
                        AddBytes(4, 0);   // input offset
                        AddBytes(4, 0);   // input location
                        AddBytes(4, conv_pool_stage.BiasDataType);   // 0x120
                        AddBytes(4, conv_pool_stage.BiasOrder);

                    }
                    else   // add or mult
                    {
                        // 2nd input info , same as first except buffer offset and location
                        AddBytes(4, it->getInputTensor(0)->getShape()[0]);  // input X-dimension size
                        AddBytes(4, it->getInputTensor(0)->getShape()[1]);  // input Y-dimension size
                        AddBytes(4, it->getInputTensor(0)->getShape()[2]);  // input Z-dimension size   (0x90)
                        AddBytes(4, blob_stats.tensor_number_size*it->getInputTensor(0)->getShape()[2]);    // InputStrideX
                        AddBytes(4, blob_stats.tensor_number_size*it->getInputTensor(0)->getShape()[2]*it->getInputTensor(0)->getShape()[0]);  // InputStrideY
                        AddBytes(4, blob_stats.tensor_number_size); // InputStrideZ
 
                        AddBytes(4, conv_pool_stage.Input1Offset);      // 2nd input
                        AddBytes(4, conv_pool_stage.Input1Location);    // 2nd Inputr
                        AddBytes(4, conv_pool_stage.OutputDataType);
                        AddBytes(4, conv_pool_stage.OutputOrder);
                    }

                    AddBytes(4, 0x5);    //  preop
                    AddBytes(4, 0x5);    //  postop

                }

            }

            uint32_t buffer_section_offset = align(next_offset,0x10) ;
            uint32_t stage_pad_size = buffer_section_offset - next_offset  ;
            AddBytes(stage_pad_size, 0x00000000);

//            std::cout << "Finished writing stages" << std::endl;
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

            for (mv::Control::OpDFSIterator it = cm.getFirst(); it != cm.opEnd(); ++it)
            {
                if (( it->getOpType() == OpType::Conv2D ) || ( it->getOpType() == OpType::FullyConnected ))
                {
                    // buffer data section for convolution has 3 regions: taps, bias, and params
                    // size of TAP region = align((roundUp(8,#kernels)*kernelX*kernelY*kernelZ)*dataSize),0x40)

                    // TAPS region
                    // calculate buffer sizes etc related to weights
                    uint32_t kernel_sizeX = 0 ;
                    uint32_t kernel_sizeY = 1 ;
                    uint32_t kernel_sizeZ = 1 ;
                    uint32_t kernel_sizeN = 1 ;

                    if ( it->getOpType() == OpType::Conv2D )
                    {
                        kernel_sizeX = it->getInputTensor(1)->getShape()[0] ;
                        kernel_sizeY = it->getInputTensor(1)->getShape()[1] ;
                        kernel_sizeZ = it->getInputTensor(1)->getShape()[2] ;
                        kernel_sizeN = it->getInputTensor(1)->getShape()[3] ;
                    }
                    else    //fc
                    {
                        kernel_sizeX = it->getInputTensor(1)->getShape().totalSize();
                    }

//                    std::cout << "this weights shape = " << kernel_sizeX << " " << kernel_sizeY << " " << kernel_sizeZ << " " << kernel_sizeN << std::endl;
                    uint32_t weights_number_size = 2 ;          // TODO assume FP16
                    uint32_t buffer_taps_weights_len = kernel_sizeX*kernel_sizeY*kernel_sizeZ*kernel_sizeN;
                    uint32_t buffer_taps_weights_size = buffer_taps_weights_len*blob_stats.weights_number_size;
                    uint32_t weights_region_size = align8(kernel_sizeN)*kernel_sizeX*kernel_sizeY*kernel_sizeZ*blob_stats.weights_number_size ;
                    weights_region_size = align(weights_region_size,64) ;
                    uint32_t weights_region_pad_size = weights_region_size - buffer_taps_weights_size ;

                    // write weights and pad to file
                    for (unsigned i=0; i< buffer_taps_weights_len; i++)
                    {
                        uint16_t cur_weight = f32Tof16(it->getInputTensor(1)->getData()[i]) ;  // TODO assume fp16
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

                    if (it->hasAttr("bias"))
                    {
                        buffer_bias_values_len = it->getAttr("bias").getContent<mv::dynamic_vector<float>>().size() ;
                        for (unsigned i = 0; i < buffer_bias_values_len; ++i)
                        {
                            buffer_bias_val = f32Tof16( it->getAttr("bias").getContent<mv::dynamic_vector<float>>()[i] );
                            AddBytes(bias_number_size, buffer_bias_val);
                        }
                    }
                    else
                    {
                        for (unsigned i=0; i< buffer_bias_values_len; i++)
                        {
                            AddBytes(bias_number_size, buffer_bias_val);
                        }
                    }

                    uint32_t buffer_bias_values_size = buffer_bias_values_len*bias_number_size;
                    uint32_t buffer_bias_region_size = align(buffer_bias_values_size,64) ;
                    uint32_t buffer_bias_pad_size = buffer_bias_region_size - (buffer_bias_values_size);

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
                }  //  end conv or FC  case
                else if ( it->getOpType() == OpType::Scale ) // scale vector 
                {
                    // BIAS region
                    uint32_t bias_number_size = 2 ;             // TODO assume FP16
                    uint16_t buffer_bias_val = f32Tof16(0.0f);  // TODO bias = 0 hardcoded
                    uint32_t buffer_bias_values_len = it->getInputTensor(1)->getShape().totalSize();

                    for (unsigned i = 0; i < buffer_bias_values_len; ++i)
                    {
                        buffer_bias_val = f32Tof16(it->getInputTensor(1)->getData()[i]);
                        AddBytes(bias_number_size, buffer_bias_val);
//                        std::cout << "scale index value " << i << " " << buffer_bias_val << std::endl;
                    }

                    uint32_t buffer_bias_values_size = buffer_bias_values_len*bias_number_size;
                    uint32_t buffer_bias_region_size = align(buffer_bias_values_size,64) ;
                    uint32_t buffer_bias_pad_size = buffer_bias_region_size - (buffer_bias_values_size);

                    for (unsigned i=0; i< buffer_bias_pad_size; i++)
                    {
                        AddBytes(1, buffer_pad_val);
                    }
                }   // end scale case
            }
       }

       void write_relocation_section(mv::ControlModel& cm)
       {
            uint32_t relocation_section_header_size = 20 ;
            uint32_t blob_buffer_reloc_size = 8*blob_stats.conv_count ;
            uint32_t work_buffer_reloc_size = 0x10 * (blob_stats.stage_count-1);
            uint32_t blob_buffer_reloc_offset = blob_stats.blob_file_size - blob_stats.relocation_section_size + relocation_section_header_size ;
            uint32_t work_buffer_reloc_offset = blob_buffer_reloc_offset + blob_buffer_reloc_size ;

            // write relocation section header
            AddBytes(4, blob_stats.relocation_section_size );
            AddBytes(4, blob_buffer_reloc_offset);
            AddBytes(4, blob_buffer_reloc_size);
            AddBytes(4, work_buffer_reloc_offset);
            AddBytes(4, work_buffer_reloc_size);

            // write buffer data relocation info
            uint32_t running_offset = 0 ;
            uint32_t node_index = 0 ;

            for (mv::Control::OpDFSIterator it = cm.getFirst(); it != cm.opEnd(); ++it)
            {
                if (( it->getOpType() == OpType::Conv2D ) || ( it->getOpType() == OpType::FullyConnected ))
                {
                    // calculate buffer sizes etc related to weights
                    uint32_t kernel_sizeX = 0 ;
                    uint32_t kernel_sizeY = 1 ;
                    uint32_t kernel_sizeZ = 1 ;
                    uint32_t kernel_sizeN = 1 ;
                    if ( it->getOpType() == OpType::Conv2D )
                    {
                        kernel_sizeX = it->getInputTensor(1)->getShape()[0] ;
                        kernel_sizeY = it->getInputTensor(1)->getShape()[1] ;
                        kernel_sizeZ = it->getInputTensor(1)->getShape()[2] ;
                        kernel_sizeN = it->getInputTensor(1)->getShape()[3] ;
                    }
                    else
                    {
                        kernel_sizeX = it->getInputTensor(1)->getShape().totalSize() ;
                    }

                    uint32_t bias_region_size = 64 ;
                    uint32_t params_region_size = 64 ;
                    uint32_t weights_region_size = align8(kernel_sizeN)*kernel_sizeX*kernel_sizeY*kernel_sizeZ*blob_stats.weights_number_size ;
                    weights_region_size = align(weights_region_size,64) ;
                    // relocation section: blob buffer relocation information
                    // weights region
                    AddBytes(4, running_offset);  // offset from start of buffer section
                    AddBytes(4, 0x00000003);          // memory type = blob-buffer
                    running_offset += weights_region_size ;
                    // bias region offset
                    AddBytes(4, running_offset);
                    AddBytes(4, 0x00000003);          // memory type = blob-buffer
                    running_offset += bias_region_size + params_region_size ;

                }   // end convolution case

                if ( it->getOpType() == OpType::Scale )
                {
                    // bias region offset
                    AddBytes(4, running_offset);
                    AddBytes(4, 0x00000003);          // memory type = blob-buffer
                    running_offset += 64 ;  
                } 
                node_index++;

           }  // end graph pass to output wts,bias buffer info

           // output work buffer relocation table
           for (unsigned j=0; j<blob_stats.relocbuf_list.size(); j++)
           {
                // relocation section: work buffer relocation information
                AddBytes(4, blob_stats.relocadr_list[j]);          // offset from start of work section
                AddBytes(4, blob_stats.relocbuf_list[j]);          // memory type =

//                cm.logger().log(mv::Logger::MessageType::MessageInfo, "writing reloc table j adr buf  = " + mv::Printable::toString(j) );

//                std::cout << "writing reloc table j adr buf  = " << j << " " << blob_stats.relocadr_list[j] << " " << blob_stats.relocbuf_list[j] << std::endl;
            }    // end loop for work buffer output

        }     // end class blob_buffer::write_relocation_section

    };   // end class blob_buffer

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

        mv::pass::FuseReLU fuseRelu;
        mv::pass::FuseScale fuseScale;
        mv::pass::FuseBias fuseBias;
        mv::pass::FuseBatchNorm fuseBatchNorm;

            uint64_t fsize = 0 ;
            switch( output_format )
            {
                case mvblob_mode:
                    // fuse relu, bias and batchnorm as required by blob
                    fuseBatchNorm.run(graph_2_deploy);
                    fuseBias.run(graph_2_deploy);
                    fuseScale.run(graph_2_deploy);
                    fuseBias.run(graph_2_deploy);
                    fuseRelu.run(graph_2_deploy);
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
