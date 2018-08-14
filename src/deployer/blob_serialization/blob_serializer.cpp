#include "include/mcm/deployer/serializer.hpp"
// #include "include/mcm/deployer/blob_serializer.hpp"
#include <stdio.h>

namespace mv
{
    void Blob_buffer::calc(mv::ControlModel& cm)
    {
        /*
            Does a soft run through to calculate all offsets for use in blob.
        */

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
        blob_stats.stage_count = 1 ;     // start count including NoOp stage
        blob_stats.data_buffer_count = 0 ;
        blob_stats.elt_count = 0 ;
        blob_stats.stage_section_size = 4*3 + 4*5 ;    // start count including 12 byte header and NoOp stage
        blob_stats.weights_region_size = 0 ;
        blob_stats.bias_region_size = 0 ;

        int additional_buf = 0;

        for (mv::Control::OpDFSIterator it = cm.getFirst(); it != cm.opEnd(); ++it)
        {
            switch(it->getOpType()){

                // case OpType::Input:
                //     {
                //         blob_stats.stage_count++;
                //     }
                // case OpType::Output:
                //     {
                //         // Pass - Not Written
                //     }
                case OpType::Conv2D:
                case OpType::FullyConnected:
                    {
                        uint32_t kernel_sizeX = 0 ;
                        uint32_t kernel_sizeY = 0 ;
                        uint32_t kernel_sizeZ = 0 ;
                        uint32_t kernel_sizeN = 0 ;

                        if ( it->getOpType() == OpType::FullyConnected )
                        {
                            kernel_sizeX = it->getInputTensor(1)->getShape().totalSize() ;
                            kernel_sizeY = 1 ;
                            kernel_sizeZ = 1 ;
                            kernel_sizeN = 1 ;
                            blob_stats.stage_section_size += (45*4) ;
                        }
                        else
                        {

                            kernel_sizeX = it->getInputTensor(1)->getShape()[0] ;
                            kernel_sizeY = it->getInputTensor(1)->getShape()[1] ;
                            kernel_sizeZ = it->getInputTensor(1)->getShape()[2] ;
                            kernel_sizeN = it->getInputTensor(1)->getShape()[3] ;


                            int mx_valid = 1;
                            if (! it->hasAttr("NCE1_Compatible"))
                            {
                                printf("Warning: attribute NCE1_Compatible not present. Assuming True.\n");
                            }
                            else
                            {

                                mx_valid = it->getAttr("NCE1_Compatible").getContent<int>();
                                mx_valid = 1;
                            }

                            if(mx_valid){

                                int descriptors = 1;
                                if (! it->hasAttr("NCE1_DescriptorSplits"))
                                {
                                    printf("Warning: attribute NCE1_DescriptorSplits not present. Defaulting to 1.\n");
                                }
                                else
                                {
                                    descriptors = it->getAttr("NCE1_DescriptorSplits").getContent<int>();
                                }
                                blob_stats.stage_section_size += (11*4) ; // Header of Descriptors
                                blob_stats.stage_section_size += (descriptors*32*4) ; // Descriptor
                                blob_stats.stage_section_size += (5*10*4) ; // Input, Bias, Taps, Output, Scale

                                blob_stats.stage_section_size += (2*4) ; // PreOp PostOp TODO: Move OUT.
                                blob_stats.stage_section_size += (3*4) ; // nextsatge, etc MOVE OUT.
                                additional_buf++;       // Has scale also
                            }else{
                                blob_stats.stage_section_size += (53*4) ;
                            }
                        }

                        // buffer data section for convolution has 3 regions: taps, bias, and params
                        // size of TAP region = align((roundUp(8,#kC)*kernelX*kernelY*kN)*dataSize),0x40)
                        //  TODO       BIAS region = align((#biases*dataSize),0x40)

                        // TAPS region
                        // calculate buffer sizes etc related to weights
                        uint32_t weights_region_size = kernel_sizeN*kernel_sizeX*kernel_sizeY*kernel_sizeZ*blob_stats.weights_number_size;
                        blob_stats.weights_region_size += weights_region_size ;
                        blob_stats.data_buffer_count++ ;

                        // calculate buffer size related to bias
                        if (it->hasAttr("bias"))
                        {
                            uint32_t buffer_bias_values_len = it->getAttr("bias").getContent<mv::dynamic_vector<float>>().size() ;
                            blob_stats.bias_region_size += buffer_bias_values_len*blob_stats.weights_number_size;
                            blob_stats.data_buffer_count++ ;
                        }

                        blob_stats.stage_count++ ;
                        if (it->hasAttr("postOpType"))
                        {
                            if (it->getAttr("postOpType").getContent<mv::OpType>() == mv::OpType::ReLU)
                            {
                                blob_stats.stage_section_size += (3*4) ;
                            }
                        }
                    }
                    break;
                case OpType::MaxPool2D:
                case OpType::AvgPool2D:
                    {
                        blob_stats.stage_count++ ;
                        blob_stats.stage_section_size += (3+7+20+2)*4 ;
                    }
                    break;
                case OpType::Add:
                case OpType::Multiply:
                    {
                        blob_stats.stage_count++ ;
                        blob_stats.elt_count++ ;
                        blob_stats.stage_section_size += (3+32)*4 ;
                    }
                    break;
                case OpType::Softmax:
                    {
                        blob_stats.stage_count++ ;
                        blob_stats.stage_section_size += (3+21+2)*4 ;
                    }
                    break;

                case OpType::ReLU:
                    {
                        blob_stats.stage_count++ ;
                        blob_stats.stage_section_size += (3+3+20+2)*4 ;
                    }
                    break;
                case OpType::Scale:
                    {
                        blob_stats.stage_count++ ;
                        blob_stats.stage_section_size += (3+32+10)*4 ;
                        blob_stats.data_buffer_count++ ;   // uses buffer section (ala wts bias)
                        uint32_t buffer_bias_values_len = ( it->getInputTensor(1)->getShape().totalSize() ) *blob_stats.weights_number_size;
                        blob_stats.bias_region_size += buffer_bias_values_len ;
                    }
                    break;
                case OpType::Conversion:
                    {
                        blob_stats.stage_count++ ;
                        blob_stats.stage_section_size += (3+20+2)*4 ;
                    }
                    break;
                default:
                    std::cout << "Calc: NO SUCH LAYER:" << Printable::toString(it->getOpType()) << std::endl;
                    assert (0);
                    break;
            }
        }

        blob_stats.output_size = cm.getLast()->getInputTensor(0)->getShape().totalSize();
        blob_stats.stage_section_size = align(blob_stats.stage_section_size, 16) ;

        // Calculate Buffer Size
        mv::DataModel dm(cm);
        mv::Control::StageIterator stg = cm.getStage(0);

        unsigned int totalSize = 0;
        unsigned int amount_const = 0;

        for(Data::BufferIterator bit = dm.bufferBegin("ConstantMemory", stg); bit != dm.bufferEnd("ConstantMemory", stg); ++bit){
            totalSize += bit->size;
            int adjustment = 0;
            while((bit->size*2 + adjustment*2) % 64 != 0){
                AddBytes(2, 0);
                adjustment++;
            }
            totalSize += adjustment;
        }

        blob_stats.buffer_data_size = totalSize*2;
        // blob_stats.buffer_data_size = blob_stats.weights_region_size + blob_stats.bias_region_size ;
        // uint32_t aligned_buffer_data_size = align(blob_stats.buffer_data_size, 64) ;
        // blob_stats.buffer_data_pad_size = aligned_buffer_data_size - blob_stats.buffer_data_size ;
        // blob_stats.buffer_data_size = aligned_buffer_data_size ;
        // blob_stats.relocation_section_size = 20 + 8*blob_stats.data_buffer_count + 16*(blob_stats.stage_count-2) + (8*blob_stats.elt_count) ;

        blob_stats.relocation_section_size = 20 + 8*blob_stats.data_buffer_count + 16*(blob_stats.stage_count-2) + (8*blob_stats.elt_count) + additional_buf*8;

        blob_stats.blob_file_size = headers_data_size+blob_stats.header_pad_size+blob_stats.stage_section_size+blob_stats.buffer_header_size+blob_stats.buffer_data_size+blob_stats.relocation_section_size ;
    }

    void Blob_buffer::write_elf_header()
    {
        /*
            Somewhat following the ELF Header standard, but effort was dropped.
            This section remains until properly depreciated.

            @param write_or_query - 0 to return size of section, 1 to write section.

        */

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
    }

    void Blob_buffer::write_mv_header()
    {

        uint32_t mv_magic_number = BLOB_MAGIC_NUMBER ;
        uint32_t mv_version_major = BLOB_VERSION_MAJOR ;
        uint32_t mv_version_minor = BLOB_VERSION_MINOR ;
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

    void Blob_buffer::write_stage_section_header()
    {
        AddBytes(4, blob_stats.stage_count);   // 0x50
        AddBytes(4, blob_stats.stage_section_size);
        AddBytes(4, blob_stats.output_size);
        AddBytes(4, 0x20);     // include input NoOp stage for compatibility with python compiler
        AddBytes(4, 0x05);
        AddBytes(4, 0x80000000);
        AddBytes(4, 0x05);
        AddBytes(4, 0x05);
    }

    void Blob_buffer::add_stage_IO_info(mv::Control::OpDFSIterator it, mv::Blob_stage conv_pool_stage)
    {
        /*
            Write Input and Output to the blob.

            - Op to pull details from
            - Blob Stage to pull details from

        */

        int inputLocation = conv_pool_stage.InputLocation;
        if (conv_pool_stage.InputLocation > 4)
        {
            inputLocation = 0x04;
        }
        int outputLocation = conv_pool_stage.OutputLocation;
        if (conv_pool_stage.OutputLocation > 4)
        {
            outputLocation = 0x04;
        }

        Blob_Tensor input = Blob_Tensor(
            it->getInputTensor(0)->getShape()[0],   // X
            it->getInputTensor(0)->getShape()[1],   // Y
            it->getInputTensor(0)->getShape()[2],   // Z
            blob_stats.tensor_number_size*it->getInputTensor(0)->getShape()[2],     // X Stride
            blob_stats.tensor_number_size*it->getInputTensor(0)->getShape()[2]*it->getInputTensor(0)->getShape()[0],    // Y Stride
            blob_stats.tensor_number_size,   // Z Stride
            conv_pool_stage.InputOffset,
            inputLocation,
            conv_pool_stage.InputDataType,
            conv_pool_stage.InputOrder
        );
        Blob_Tensor output = Blob_Tensor(
            it->getOutputTensor(0)->getShape()[0],   // X
            it->getOutputTensor(0)->getShape()[1],   // Y
            it->getOutputTensor(0)->getShape()[2],   // Z
            blob_stats.tensor_number_size*it->getOutputTensor(0)->getShape()[2],     // X Stride
            blob_stats.tensor_number_size*it->getOutputTensor(0)->getShape()[2]*it->getOutputTensor(0)->getShape()[0],    // Y Stride
            conv_pool_stage.OutputStrideZ,   // Z Stride
            conv_pool_stage.OutputOffset,
            outputLocation,
            conv_pool_stage.OutputDataType,
            conv_pool_stage.OutputOrder
        );

        input.write(this);
        output.write(this);
    }

    void Blob_buffer::write_stages(mv::ControlModel& cm)
    {

        Blob_stage conv_pool_stage ;
        uint32_t op_count = 0 ;
        uint32_t next_offset = 4*3 + 4*5 ;
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

            Blob_Op_Definition op_spec = Blob_Op_Definition(it->getOpType());

            if (op_spec.number_of_inputs == 1)
            {
                // determine source
                auto parentIt = om.getSourceOp(it->getInputTensor(0));

                if (parentIt->getOpType() == OpType::Input)
                {
                    inbufnum_list.push_back(1);
                    sourcename_list.push_back("Input");
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
                            //std::cout << "pushing inbuffer_list (branch input) "<< work_buffer_index-1 << " " << parentIt->getName() << std::endl;
                        }
                    }
                    if (!branch_input)    // new buffer needed
                    {
                        inbufnum_list.push_back(work_buffer_index++);
                        sourcename_list.push_back(parentIt->getName());
                        workbuffer_offsets.push_back(1);   // create unsized buffer position at index=num-4
                        //std::cout << "pushing inbuffer_list "<< work_buffer_index-1 << " " << parentIt->getName() << std::endl;
                        //std::cout << "   WBO_list size = "<<  workbuffer_offsets.size() << std::endl;
                    }
                }
            } // end single input operator case

            else if (op_spec.number_of_inputs == 2)
            {

                for ( int input_index = 0; input_index <2; input_index++ )
                {
                    // determine source 0
                    auto parentIt = om.getSourceOp(it->getInputTensor(input_index));

                    if (parentIt->getOpType() == OpType::Input)
                    {
                        inbufnum_list.push_back(1);
                        sourcename_list.push_back("Input");
                        //std::cout << "pushing inbuffer_list 1 Input 0" << std::endl;
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
                                //std::cout << "pushing inbuffer_list (branch input) "<< common_node << " " << parentIt->getName() << std::endl;
                            }
                        }
                        if (!branch_input)    // new buffer needed
                        {
                            inbufnum_list.push_back(work_buffer_index++);
                            sourcename_list.push_back(parentIt->getName());
                            workbuffer_offsets.push_back(1);   // create unsized buffer position at index=num-4
                            //std::cout << "pushing inbuffer_list "<< work_buffer_index-1 << " " << parentIt->getName() << std::endl;
                            //std::cout << "   WBO_list size = "<<  workbuffer_offsets.size() << std::endl;
                        }
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
                }
                else
                {
                    inbufnum_list.push_back(2);
                    sourcename_list.push_back(parentIt->getName());
                    workbuffer_offsets.push_back(1);   // create unsized buffer position at index=num-4
                }
            } // end output node case
            else if (it->getOpType() == OpType::Input)
            {
            }
            else{
                printf("Warning: No buffers associated with Layer.\n");
            }

        }    // end input buffer calculation pass

        // traverse graph to determine output buffer number and size for each node in the computation
        // buffer numbers retreived from input buffer list with matching source name
        // store size and buffer number for later
        uint32_t running_offset = 0 ;
        for (mv::Control::OpDFSIterator it = cm.getFirst(); it != cm.opEnd(); ++it)
        {

            //Blob_Op_Definition op_spec = Blob_Op_Definition(it->getOpType());

            int work_buffer_size = 0;
            if (it->getOpType() != OpType::Output)
            {
                int padX = 0;
                int padY = 0;

                if (it->getOpType() == OpType::Conv2D)
                {
                    padX = ((((it->getInputTensor(1)->getShape()[0])/2)+1)*2) ;   // compatibility pad allowing conv output overrun
                    padY = 0;
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
                        if (inbufnum_list[list_index]>=4)
                        {
                            if (workbuffer_offsets[inbufnum_list[list_index]-4] == 1)
                            {
                                outbufnum_list.push_back(inbufnum_list[list_index]);
                                outbufsiz_list.push_back(work_buffer_size);
                                workbuffer_offsets[inbufnum_list[list_index]-4]=running_offset;
                                running_offset += work_buffer_size;
                            }
                        }
                        else if (inbufnum_list[list_index]==2)
                        {
                            outbufnum_list.push_back(2);
                            outbufsiz_list.push_back(0);
                        }
                    }
                }   // end search inbuflist for match
            }   // end not-output case (no output tensor from output node)
        }   // end pass to fill outbuf lists

        // calculate address offset for each work buffer in inbufnum_list
        // find buffer size from outbufsiz_list
        for ( uint32_t inbuf_index = 0; inbuf_index < inbufnum_list.size(); inbuf_index++ )
        {
            uint32_t bufr2size = inbufnum_list[inbuf_index];
            if ( bufr2size >= 4 )
            {
                inbufadr_list.push_back(workbuffer_offsets[bufr2size-4]);
            }     // end if WORK buffer
            else
            {
                inbufadr_list.push_back(0);
            }
        }   // end inbuflist loop

        //  fill outbufadr_list
        for ( uint32_t obuf_index = 0; obuf_index < outbufnum_list.size(); obuf_index++ )
        {
            uint32_t bufr2copy = outbufnum_list[obuf_index];
            if (bufr2copy >= 4)
            {
                outbufadr_list.push_back(workbuffer_offsets[bufr2copy-4]);
            }     // end if WORK buffer
            else
            {
                outbufadr_list.push_back(0);
            }
        }   // end outbuf list loop

        // pass to output stage info -----------------------------------
        int outlist_index = 0 ;
        int inlist_index = 0 ;
        int reloc_index = 0 ;
        for (mv::Control::OpDFSIterator it = cm.getFirst(); it != cm.opEnd(); ++it)
        {

            switch(it->getOpType()){
                case OpType::Conv2D:
                    {
                        int mx_valid = 1;
                        if (! it->hasAttr("NCE1_Compatible"))
                        {
                            printf("Warning: attribute NCE1_Compatible not present. Assuming True.\n");
                        }
                        else
                        {
                            mx_valid = it->getAttr("NCE1_Compatible").getContent<int>();
                            printf("Compatibility Flag from Compiler: %i\n", mx_valid);
                            mx_valid = 1;
                        }

                        if(mx_valid)
                        {


                            int descriptors = 1;
                            int point0 = 0;
                            if (! it->hasAttr("NCE1_DescriptorSplits"))
                            {
                                printf("Warning: attribute NCE1_DescriptorSplits not present. Defaulting to 1.\n");
                            }
                            else
                            {
                                descriptors = it->getAttr("NCE1_DescriptorSplits").getContent<int>();
                            }
                            point0 += (11*4) ; // Header of Descriptors
                            point0 += (descriptors*32*4) ; // Descriptor
                            point0 += (5*10*4) ; // Input, Bias, Taps, Output, Scale

                            point0 += (2*4) ; // PreOp PostOp TODO: Move OUT.
                            point0 += (3*4) ; // nextsatge, etc MOVE OUT.

                            next_offset += point0 ;

                            conv_pool_stage.next = next_offset;

                            AddBytes(4, conv_pool_stage.next);
                            AddBytes(4, 0x21);                                // 0x60
                            AddBytes(4, conv_pool_stage.implementation);

                            // Serialize for MyriadX H/W
                            bConv2D c = bConv2D(&(*it));
                            c.writeStageInfo(&om, this);

                            AddBytes(4, 0x05);    // 0x12c , no preop
                            AddBytes(4, 0x05);    // 0x12c , no postop
                        }
                        else
                        {
                            // Serialize for S/W

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

                            Blob_Tensor taps = Blob_Tensor(
                                it->getInputTensor(1)->getShape()[0]*it->getInputTensor(1)->getShape()[1],  // X
                                it->getInputTensor(1)->getShape()[2],   // y
                                it->getInputTensor(1)->getShape()[3],   // z
                                blob_stats.tensor_number_size*it->getInputTensor(1)->getShape()[2]*it->getInputTensor(1)->getShape()[3],
                                blob_stats.tensor_number_size*it->getInputTensor(1)->getShape()[3], // Taps Sy
                                conv_pool_stage.TapsStrideZ, // SZ
                                conv_pool_stage.TBOffset,
                                conv_pool_stage.TapsLocation,
                                conv_pool_stage.TapsDataType,
                                conv_pool_stage.TapsOrder
                            );
                            conv_pool_stage.TBOffset++ ;
                            taps.write(this);


                            int biasOffset = 0, biasLocation = 0, biasDataType = 0, biasOrder = 0;

                            if (it->hasAttr("bias"))
                            {
                                conv_pool_stage.BiasDimX = it->getAttr("bias").getContent<mv::dynamic_vector<float>>().size() ;
                                conv_pool_stage.BiasStrideY = conv_pool_stage.BiasStrideX*conv_pool_stage.BiasDimX;
                                conv_pool_stage.BiasStrideZ = conv_pool_stage.BiasStrideY;
                                biasOffset = conv_pool_stage.TBOffset;
                                conv_pool_stage.TBOffset++ ;
                                biasLocation = conv_pool_stage.BiasLocation;
                                biasDataType = conv_pool_stage.BiasDataType;
                                biasOrder = 1;  // TODO: should not be hardcoded
                            }

                            Blob_Tensor bias = Blob_Tensor(
                                conv_pool_stage.BiasDimX,  // X
                                conv_pool_stage.BiasDimY,   // y
                                conv_pool_stage.BiasDimZ,   // z
                                conv_pool_stage.BiasStrideX,
                                conv_pool_stage.BiasStrideY,
                                conv_pool_stage.BiasStrideZ,
                                biasOffset,
                                biasLocation,
                                biasDataType,
                                biasOrder   // Order
                            );
                            bias.write(this);

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
                                else
                                {
                                    std::cout << "ERROR: NON-relu postOP found for " << it->getName() << std::endl;
                                }

                            }
                            else
                            {
                                if (it->hasAttr("bias"))
                                {
                                    AddBytes(4, 0x09);    // 0x12c , postop bias
                                }
                                else
                                {
                                    AddBytes(4, 0x05);    // 0x12c , no postop
                                }
                            }
                        }
                    }
                    break;
                case OpType::FullyConnected:
                    {
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

                        Blob_Tensor input = Blob_Tensor(
                            it->getInputTensor(0)->getShape()[0],   // X
                            it->getInputTensor(0)->getShape()[1],   // Y
                            it->getInputTensor(0)->getShape()[2],   // Z
                            blob_stats.tensor_number_size*it->getInputTensor(0)->getShape()[2],     // X Stride
                            blob_stats.tensor_number_size*it->getInputTensor(0)->getShape()[2]*it->getInputTensor(0)->getShape()[0],    // Y Stride
                            blob_stats.tensor_number_size,   // Z Stride
                            conv_pool_stage.InputOffset,
                            conv_pool_stage.InputLocation,
                            conv_pool_stage.InputDataType,
                            conv_pool_stage.InputOrder
                        );

                        Blob_Tensor output = Blob_Tensor(
                            0x01,   // X
                            0x01,   // Y
                            it->getOutputTensor(0)->getShape().totalSize(),   // Z
                            blob_stats.tensor_number_size*it->getOutputTensor(0)->getShape().totalSize(),     // X Stride
                            blob_stats.tensor_number_size*it->getOutputTensor(0)->getShape().totalSize(),    // Y Stride
                            blob_stats.tensor_number_size,
                            conv_pool_stage.OutputOffset,
                            conv_pool_stage.OutputLocation,
                            conv_pool_stage.OutputDataType,
                            conv_pool_stage.OutputOrder
                        );
                        Blob_Tensor bias = Blob_Tensor(
                            it->getOutputTensor(0)->getShape().totalSize(),   // X
                            0x01,   // Y
                            0x01,   // Z
                            blob_stats.tensor_number_size,     // X Stride
                            blob_stats.tensor_number_size*it->getOutputTensor(0)->getShape().totalSize(),    // Y Stride
                            blob_stats.tensor_number_size*it->getOutputTensor(0)->getShape().totalSize(),    // z Stride
                            conv_pool_stage.TBOffset,
                            conv_pool_stage.BiasLocation,
                            conv_pool_stage.BiasDataType,
                            conv_pool_stage.BiasOrder
                        );
                        conv_pool_stage.TBOffset++ ;


                        input.write(this);
                        output.write(this);
                        bias.write(this);

                        AddBytes(4, conv_pool_stage.preop_type);
                        if (it->hasAttr("postOpType"))
                        {
                            if (it->getAttr("postOpType").getContent<mv::OpType>() == mv::OpType::ReLU)
                            {
                                //std::cout << "--relu found for " << it->getName() << std::endl;
                                AddBytes(4, 0x06);    // 0x12c , postop relu
                                AddBytes(4, 0x00);
                                AddBytes(4, 0x00);
                                AddBytes(4, 0x00);
                            }
                            else
                            {
                                //std::cout << "ERROR: NON-relu postOP found for " << it->getName() << std::endl;
                            }
                        }
                        else
                        {
                            if (it->hasAttr("bias"))
                            {
                                //std::cout << "--bias found for " << it->getName() << std::endl;
                                AddBytes(4, 0x09);    // 0x12c , postop bias
                            }
                            else
                            {
                                //std::cout << "--no postop attr for " << it->getName() << std::endl;
                                AddBytes(4, 0x05);    // 0x12c , no postop
                            }
                        }
                    }
                    break;
                case OpType::Softmax:
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
                                //std::cout << "pushing reloc-table (softmax in)relindex bufnum siz "<< reloc_index << " " <<  outbufnum_list[olist_index] << " " << outbufsiz_list[olist_index] << std::endl;
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
                                //std::cout << "pushing reloc-table (softmax out) relindex bufnum siz "<< reloc_index << " " <<  outbufnum_list[outlist_index] << " " << outbufsiz_list[outlist_index] << std::endl;
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


                        int inputLocation = conv_pool_stage.InputLocation;
                        if (conv_pool_stage.InputLocation > 4)
                        {
                            inputLocation = 0x04;
                        }
                        int outputLocation = conv_pool_stage.OutputLocation;
                        if (conv_pool_stage.OutputLocation > 4)
                        {
                            outputLocation = 0x04;
                        }

                        Blob_Tensor input = Blob_Tensor(
                            0x01,   // X
                            0x01,   // Y
                            it->getInputTensor(0)->getShape().totalSize(),   // Z
                            blob_stats.tensor_number_size*it->getInputTensor(0)->getShape().totalSize(),     // X Stride
                            blob_stats.tensor_number_size*it->getInputTensor(0)->getShape().totalSize(),    // Y Stride
                            blob_stats.tensor_number_size,
                            conv_pool_stage.InputOffset,
                            inputLocation,
                            conv_pool_stage.InputDataType,
                            conv_pool_stage.InputOrder
                        );
                        Blob_Tensor output = Blob_Tensor(
                            0x01,   // X
                            0x01,   // Y
                            it->getOutputTensor(0)->getShape().totalSize(),   // Z
                            blob_stats.tensor_number_size*it->getOutputTensor(0)->getShape().totalSize(),     // X Stride
                            blob_stats.tensor_number_size*it->getOutputTensor(0)->getShape().totalSize(),    // Y Stride
                            blob_stats.tensor_number_size,
                            conv_pool_stage.OutputOffset,
                            outputLocation,
                            conv_pool_stage.OutputDataType,
                            conv_pool_stage.OutputOrder
                        );
                        input.write(this);
                        output.write(this);

                        AddBytes(4, conv_pool_stage.preop_type);
                        AddBytes(4, conv_pool_stage.postop_type);
                    }
                    break;
                case OpType::ReLU:
                    {
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
                                //std::cout << "pushing reloc-table (relu in) relindex bufnum siz "<< reloc_index << " " <<  outbufnum_list[olist_index] << " " << outbufsiz_list[olist_index] << std::endl;
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

                        add_stage_IO_info(it, conv_pool_stage);

                        AddBytes(4, 0x00); // post stride x
                        AddBytes(4, 0x00); // post stride y

                        AddBytes(4, conv_pool_stage.preop_type);
                        AddBytes(4, conv_pool_stage.postop_type);
                    }
                    break;
                case OpType::MaxPool2D:
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
                        AddBytes(4, conv_pool_stage.implementation);

                        // operator specific info
                        AddBytes(4, it->getAttr("kSize").getContent<mv::UnsignedVector2D>().e0); // radix X
                        AddBytes(4, it->getAttr("kSize").getContent<mv::UnsignedVector2D>().e1); // radix Y (0x140)
                        AddBytes(4, it->getAttr("stride").getContent<mv::UnsignedVector2D>().e0); //strideX
                        AddBytes(4, it->getAttr("stride").getContent<mv::UnsignedVector2D>().e1); //strideY
                        AddBytes(4, 0x00);   // padX
                        AddBytes(4, 0x00);   // padY 0x150
                        AddBytes(4, conv_pool_stage.padStyle);

                        add_stage_IO_info(it, conv_pool_stage);

                        AddBytes(4, conv_pool_stage.preop_type);
                        AddBytes(4, 0x05);    // 0x1ac  postop type
                    }
                    break;
                case OpType::AvgPool2D:
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
                        AddBytes(4, conv_pool_stage.implementation);

                        // operator specific info
                        AddBytes(4, it->getAttr("kSize").getContent<mv::UnsignedVector2D>().e0); // radix X
                        AddBytes(4, it->getAttr("kSize").getContent<mv::UnsignedVector2D>().e1); // radix Y (0x140)
                        AddBytes(4, it->getAttr("stride").getContent<mv::UnsignedVector2D>().e0); //strideX
                        AddBytes(4, it->getAttr("stride").getContent<mv::UnsignedVector2D>().e1); //strideY
                        AddBytes(4, 0x00);   // padX
                        AddBytes(4, 0x00);   // padY 0x150
                        AddBytes(4, conv_pool_stage.padStyle);

                        add_stage_IO_info(it, conv_pool_stage);

                        AddBytes(4, conv_pool_stage.preop_type);
                        AddBytes(4, 0x05);    // 0x1ac  postop type
                    }
                    break;
                case OpType::Add:
                case OpType::Multiply:
                case OpType::Scale:
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

                            Blob_Tensor input_2 = Blob_Tensor(
                                0x00,   // X
                                1,   // Y
                                1,   // Z
                                blob_stats.tensor_number_size,     // X Stride
                                blob_stats.tensor_number_size*it->getInputTensor(1)->getShape().totalSize(),    // Y Stride
                                blob_stats.tensor_number_size,
                                conv_pool_stage.TBOffset,
                                3,
                                conv_pool_stage.OutputDataType,
                                3
                            );
                            conv_pool_stage.TBOffset++;


                            Blob_Tensor bias = Blob_Tensor(
                                conv_pool_stage.BiasDimX,
                                conv_pool_stage.BiasDimY,
                                conv_pool_stage.BiasDimZ,
                                conv_pool_stage.BiasStrideX,
                                conv_pool_stage.BiasStrideY,
                                conv_pool_stage.BiasStrideZ,
                                0,
                                0,
                                conv_pool_stage.BiasDataType,
                                conv_pool_stage.BiasOrder
                            );

                            input_2.write(this);
                            bias.write(this);

                        }
                        else   // add or mult
                        {
                            // 2nd input info , same as first except buffer offset and location


                            int input_1Location = conv_pool_stage.Input1Location;
                            if (conv_pool_stage.Input1Location > 4)
                            {
                                input_1Location = 0x04;
                            }

                            Blob_Tensor input = Blob_Tensor(
                                it->getInputTensor(0)->getShape()[0],   // X
                                it->getInputTensor(0)->getShape()[1],   // Y
                                it->getInputTensor(0)->getShape()[2],   // Z
                                blob_stats.tensor_number_size*it->getInputTensor(0)->getShape()[2],     // X Stride
                                blob_stats.tensor_number_size*it->getInputTensor(0)->getShape()[2]*it->getInputTensor(0)->getShape()[0],    // Y Stride
                                blob_stats.tensor_number_size,
                                conv_pool_stage.Input1Offset,
                                input_1Location,
                                conv_pool_stage.OutputDataType,
                                conv_pool_stage.OutputOrder
                            );
                            input.write(this);
                        }

                        AddBytes(4, 0x5);    //  preop
                        AddBytes(4, 0x5);    //  postop
                    }
                    break;
                case OpType::Conversion:
                    {

                        int point0 = 0;
                        point0 += 2*10 ; // Input, Output
                        point0 += 2 ; // PreOp PostOp TODO: Move OUT.
                        point0 += 3 ; // nextstage, id, imp
                        next_offset += point0*4 ;


                        mv::DataModel dm(om);
                        mv::ControlModel cm(om);
                        Data::BufferIterator mem;
                        mv::Control::StageIterator stg = cm.getStage(0);
                        auto t = it->getOutputTensor(0);

                        mem = dm.getBuffer("IntermediateMemory", stg, t);

                        if (mem == dm.bufferEnd("IntermediateMemory", stg)  ){
                            next_offset = 0;
                        }

                        conv_pool_stage.next = next_offset ;
                        AddBytes(4, conv_pool_stage.next);
                        AddBytes(4, 0x25);                                // 0x60
                        AddBytes(4, conv_pool_stage.implementation);

                        // Serialize for MyriadX H/W
                        bCompatibility c = bCompatibility(&(*it));
                        c.writeStageInfo(&om, this);

                        AddBytes(4, 0x05);    // 0x12c , no preop
                        AddBytes(4, 0x05);    // 0x12c , no postop

                    }
                    break;

                default:
                    std::cout << "Write: NO SUCH LAYER:" << Printable::toString(it->getOpType()) << std::endl;
                    assert(0);
            }
        }

        uint32_t buffer_section_offset = align(next_offset,0x10) ;
        uint32_t stage_pad_size = buffer_section_offset - next_offset  ;
        if (stage_pad_size > 0)
        {
            AddBytes(stage_pad_size, 0x00000000);
        }

        //std::cout << "Finished writing stages section of blob" << std::endl;
    }

    void Blob_buffer::write_buffer_section(mv::ControlModel& cm)
    {
        uint32_t buffer_header_pad_size = 3 ;
        uint32_t buffer_header_pad_val = 0x002a ;
        uint8_t buffer_pad_val = 0x00 ;
        Float16Compressor cvtr ;

        mv::DataModel dm(cm);
        mv::Control::StageIterator stg = cm.getStage(0);

        // unsigned int totalSize = 0;

        // for(Data::BufferIterator bit = dm.bufferBegin("ConstantMemory", stg); bit != dm.bufferEnd("ConstantMemory", stg); ++bit){
        //     totalSize += bit->size;
        //     int adjustment = 0;
        //     while((bit->size*2 + adjustment*2) % 64 != 0){
        //         AddBytes(2, 0);
        //         adjustment++;
        //     }
        //     totalSize += adjustment;
        // }

        // buffer section header
        AddBytes(4, (blob_stats.buffer_header_size + blob_stats.buffer_data_size));
        // AddBytes(4, (blob_stats.buffer_header_size + totalSize*2));

        for (unsigned i=0; i<buffer_header_pad_size; i++)
        {
            AddBytes(4, buffer_header_pad_val);
        }


        for(Data::BufferIterator bit = dm.bufferBegin("ConstantMemory", stg); bit != dm.bufferEnd("ConstantMemory", stg); ++bit){
            for (int idx = 0; idx != (int)bit->size; idx++){
                u_int16_t fp16_val = cvtr.compress((*bit->data).getData()[idx]) ;  // Convert to fp16.
                AddBytes(2, fp16_val) ;
            }

            int adjustment = 0;
            while((bit->size*2 + adjustment*2) % 64 != 0){
                AddBytes(2, 0);
                adjustment++;
            }
        }
    }

    blob_summary Blob_buffer::getBlobSumm(){
        return this->blob_stats;
    }

    void Blob_buffer::write_relocation_section(mv::ControlModel& cm)
    {

        this->reloc_table.write(this);

        // uint32_t relocation_section_header_size = 20 ;
        // uint32_t blob_buffer_reloc_size = 8*blob_stats.data_buffer_count ;
        // uint32_t work_buffer_reloc_size = 0x10 * (blob_stats.stage_count-2) + 8*blob_stats.elt_count ;
        // uint32_t blob_buffer_reloc_offset = blob_stats.blob_file_size - blob_stats.relocation_section_size + relocation_section_header_size ;
        // uint32_t work_buffer_reloc_offset = blob_buffer_reloc_offset + blob_buffer_reloc_size ;

        // // write relocation section header
        // AddBytes(4, blob_stats.relocation_section_size );
        // AddBytes(4, blob_buffer_reloc_offset);
        // AddBytes(4, blob_buffer_reloc_size);
        // AddBytes(4, work_buffer_reloc_offset);
        // AddBytes(4, work_buffer_reloc_size);

        // // write buffer data relocation info
        // uint32_t running_offset = 0 ;
        // uint32_t node_index = 0 ;

        // for (mv::Control::OpDFSIterator it = cm.getFirst(); it != cm.opEnd(); ++it)
        // {
        //     if (( it->getOpType() == OpType::Conv2D ) || ( it->getOpType() == OpType::FullyConnected ))
        //     {
        //         // calculate buffer sizes etc related to weights
        //         uint32_t kernel_sizeX = 0 ;
        //         uint32_t kernel_sizeY = 1 ;
        //         uint32_t kernel_sizeZ = 1 ;
        //         uint32_t kernel_sizeN = 1 ;
        //         if ( it->getOpType() == OpType::Conv2D )
        //         {
        //             kernel_sizeX = it->getInputTensor(1)->getShape()[0] ;
        //             kernel_sizeY = it->getInputTensor(1)->getShape()[1] ;
        //             kernel_sizeZ = it->getInputTensor(1)->getShape()[2] ;
        //             kernel_sizeN = it->getInputTensor(1)->getShape()[3] ;
        //         }
        //         else
        //         {
        //             kernel_sizeX = it->getInputTensor(1)->getShape().totalSize() ;
        //         }

        //         uint32_t bias_region_size = 0 ;
        //         uint32_t bias_number_size = 2 ;             // TODO assume FP16
        //         uint32_t bias_values_len = 1;        // TODO use 1 for now (same bias all outputs)

        //         if (it->hasAttr("bias"))
        //         {
        //             bias_values_len = it->getAttr("bias").getContent<mv::dynamic_vector<float>>().size() ;
        //         }

        //         uint32_t bias_values_size = bias_values_len*bias_number_size;
        //         bias_region_size = bias_values_size ;
        //         //std::cout << "    bias region size (in reloc table) =  "<<  bias_region_size << std::endl;

        //         uint32_t weights_region_size = kernel_sizeN*kernel_sizeX*kernel_sizeY*kernel_sizeZ*blob_stats.weights_number_size ;
        //         // relocation section: blob buffer relocation information
        //         // weights region
        //         AddBytes(4, running_offset);  // offset from start of buffer section
        //         AddBytes(4, 0x00000003);          // memory type = blob-buffer
        //         running_offset += weights_region_size ;
        //         // bias region offset
        //         AddBytes(4, running_offset);
        //         AddBytes(4, 0x00000003);          // memory type = blob-buffer
        //         running_offset += bias_region_size ;

        //     }   // end convolution, FC case

        //     if ( it->getOpType() == OpType::Scale )
        //     {
        //         uint32_t bias_region_size = 0 ;
        //         uint32_t bias_number_size = 2 ;             // TODO assume FP16
        //         uint32_t bias_values_len = 1;        // TODO use 1 for now (same bias all outputs)

        //         bias_values_len = it->getAttr("bias").getContent<mv::dynamic_vector<float>>().size() ;

        //         uint32_t bias_values_size = bias_values_len*bias_number_size;
        //         bias_region_size = bias_values_size ;
        //         //std::cout << "    bias region size (in reloc table) =  "<<  bias_region_size << std::endl;

        //         // bias region offset
        //         AddBytes(4, running_offset);
        //         AddBytes(4, 0x00000003);          // memory type = blob-buffer
        //         running_offset += bias_region_size ;
        //     }
        //     node_index++;

        // }  // end graph pass to output wts,bias buffer info

        // // output work buffer relocation table
        // for (unsigned j=0; j<blob_stats.relocbuf_list.size(); j++)
        // {
        //     // relocation section: work buffer relocation information
        //     AddBytes(4, blob_stats.relocadr_list[j]);          // offset from start of work section
        //     if (blob_stats.relocbuf_list[j]>4)
        //     {
        //         blob_stats.relocbuf_list[j] = 4 ;
        //     }
        //     AddBytes(4, blob_stats.relocbuf_list[j]);          // memory type =
        // }    // end loop for work buffer output
    }
}
