#include "include/mcm/deployer/serializer.hpp"
// #include "include/mcm/deployer/blob_serializer.hpp"
#include <stdio.h>

#define FORCE_DISABLE_BIAS

namespace mv
{
    void Blob_buffer::calc(mv::ControlModel& cm)
    {
        /*
            Does a soft run through to calculate all offsets for use in blob.
        */


        std::cout << "--- Calc ---" << std::endl;

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
        int total_bias_elements = 0;

        mv::DataModel dm(cm);

        for (mv::Control::OpDFSIterator it = cm.getFirst(); it != cm.opEnd(); ++it)
        {
            switch(it->getOpType()){
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


                            int mx_valid = 0;
                            if (! it->hasAttr("NCE1_Compatible"))
                            {
                                printf("Warning: attribute NCE1_Compatible not present. Assuming False.\n");
                            }
                            else
                            {
                                mx_valid = it->getAttr("NCE1_Compatible").getContent<int>();
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
                            uint32_t buffer_bias_values_len = dm.findTensor(it->getAttr("bias").getContent<std::string>())->getData().size() ;
                            blob_stats.bias_region_size += buffer_bias_values_len*blob_stats.weights_number_size;
                            blob_stats.data_buffer_count++ ;
                            if(buffer_bias_values_len % 64 != 0){
                                // std::cout << buffer_bias_values_len << "->" << align(buffer_bias_values_len, 32) << std::endl;
                                total_bias_elements+=align(buffer_bias_values_len, 32);
                            }else{
                                total_bias_elements+=buffer_bias_values_len;
                            }
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
                        blob_stats.stage_section_size += bPooling::getSerializedSize()+5*4 ;
                    }
                    break;
                case OpType::Add:
                case OpType::Multiply:
                    {
                        blob_stats.stage_count++ ;
                        blob_stats.elt_count++ ;
                        blob_stats.stage_section_size += bEltwise::getSerializedSize()+5*4 ;
                    }
                    break;
                case OpType::Softmax:
                    {
                        blob_stats.stage_count++ ;
                        blob_stats.stage_section_size += bSoftmax::getSerializedSize()+5*4 ;
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
                        #ifndef FORCE_DISABLE_BIAS
                            blob_stats.data_buffer_count++ ;   // uses buffer section (ala wts bias)
                        #endif
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
        mv::Control::StageIterator stg = cm.getStage(0);

        unsigned int totalSize = 0;
        unsigned int amount_const = 0;

        try{
            for(Data::BufferIterator bit = dm.bufferBegin("ConstantMemory", stg); bit != dm.bufferEnd("ConstantMemory", stg); ++bit){
                totalSize += bit->size;
                int adjustment = 0;
                while((bit->size*2 + adjustment*2) % 64 != 0){
                    AddBytes(2, 0);
                    adjustment++;
                }
                totalSize += adjustment;
            }
        }catch(mv::ArgumentError){
            std::cout << "Warning: No Constant Memory Present." << std::endl;
        }

        // TODO: Bias should be part of ConstantMemory already, not have this seperate case.
        blob_stats.buffer_data_size = totalSize*2 + total_bias_elements*2;


        std::cout << "Reloc Size Calc: " << 20 << ", " << blob_stats.data_buffer_count << "," << blob_stats.stage_count-2 << ", " << blob_stats.elt_count << "," << additional_buf << std::endl;
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

        std::cout << "--- Write Stages ---" << std::endl;

        Blob_stage conv_pool_stage ;
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

            std::cout << "> " << mv::Printable::toString(it->getOpType()) << std::endl;


            switch(it->getOpType()){
                case OpType::Conv2D:
                    {
                        int mx_valid = 0;
                        if (! it->hasAttr("NCE1_Compatible"))
                        {
                            printf("Warning: attribute NCE1_Compatible not present. Assuming False.\n");
                        }
                        else
                        {
                            mx_valid = it->getAttr("NCE1_Compatible").getContent<int>();
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


                            // No more layers (last)
                            mv::DataModel dm(om);
                            mv::ControlModel cm(om);
                            Data::BufferIterator mem;
                            mv::Control::StageIterator stg = cm.getStage(0);

                            int finalstage = 0;
                            try{
                                auto t = it->getOutputTensor(0);
                                mem = dm.getBuffer("IntermediateMemory", stg, t);
                                if (mem == dm.bufferEnd("IntermediateMemory", stg)  ){
                                    conv_pool_stage.next = 0;
                                    finalstage = 1;
                                }
                            }catch(mv::ArgumentError){
                                printf("Warning: No Intermediary Buffers\n");
                                conv_pool_stage.next = 0;
                                finalstage = 1;
                            }
                            if(!finalstage){
                                conv_pool_stage.next = next_offset;
                            }

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

                            int descriptors = 1;
                            int point0 = 0;
                            point0 += (8*4) ; // Fields
                            point0 += (4*10*4) ; // Input, Bias, Taps, Output, Scale
                            point0 += (3*4) ; // nextsatge, etc MOVE OUT.


                            if (it->hasAttr("postOpType"))
                            {
                                if (it->getAttr("postOpType").getContent<mv::OpType>() == mv::OpType::ReLU)
                                {
                                    point0 += (5*4) ;
                                }else{
                                    printf("POST OP NOT SUPPORTED\n"); // TODO: Move out.
                                    assert(0);
                                }
                            }
                            else
                            {
                                point0 += (2*4) ;
                            }

                            next_offset += point0 ;


                            // No more layers (last)
                            mv::DataModel dm(om);
                            mv::ControlModel cm(om);
                            Data::BufferIterator mem;
                            mv::Control::StageIterator stg = cm.getStage(0);
                            int finalstage = 0;
                            try{
                                auto t = it->getOutputTensor(0);
                                mem = dm.getBuffer("IntermediateMemory", stg, t);
                                if (mem == dm.bufferEnd("IntermediateMemory", stg)  ){
                                    conv_pool_stage.next = 0;
                                    finalstage = 1;
                                }
                            }catch(mv::ArgumentError){
                                printf("Warning: No Intermediary Buffers\n");
                                conv_pool_stage.next = 0;
                                finalstage = 1;
                            }
                            if(!finalstage){
                                conv_pool_stage.next = next_offset;
                            }

                            AddBytes(4, conv_pool_stage.next);
                            AddBytes(4, 0);                                // 0x60
                            AddBytes(4, conv_pool_stage.implementation);

                            // Serialize for MyriadX H/W
                            bConv2D c = bConv2D(&(*it));
                            c.writeStageInfo(&om, this);

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
                        if(0){
                            int point0 = 0;
                            point0 += 4*10 ; // Input, Output
                            point0 += 2 ; // PreOp PostOp TODO: Move OUT.
                            point0 += 3 ; // nextstage, id, imp
                            next_offset += point0*4 ;

                            // No more layers (last)
                            mv::DataModel dm(om);
                            mv::ControlModel cm(om);
                            Data::BufferIterator mem;
                            mv::Control::StageIterator stg = cm.getStage(0);
                            int finalstage = 0;
                            try{
                                auto t = it->getOutputTensor(0);
                                mem = dm.getBuffer("IntermediateMemory", stg, t);
                                if (mem == dm.bufferEnd("IntermediateMemory", stg)  ){
                                    conv_pool_stage.next = 0;
                                    finalstage = 1;
                                }
                            }catch(mv::ArgumentError){
                                printf("Warning: No Intermediary Buffers\n");
                                conv_pool_stage.next = 0;
                                finalstage = 1;
                            }
                            if (!finalstage){
                                conv_pool_stage.next = next_offset;
                            }
                            AddBytes(4, conv_pool_stage.next);
                            AddBytes(4, 0x04);                                // 0x60
                            AddBytes(4, conv_pool_stage.implementation);

                            // Serialize for MyriadX H/W
                            bInnerProduct c = bInnerProduct(&(*it));
                            c.writeStageInfo(&om, this);

                            AddBytes(4, 0x05);    // 0x12c , no preop
                            AddBytes(4, 0x05);    // 0x12c , no postop
                        }else{
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

                                    #ifdef FORCE_DISABLE_BIAS
                                        AddBytes(4, 0x05);
                                    #else
                                        AddBytes(4, 0x09);    // 0x12c , postop bias
                                    #endif
                                }
                                else
                                {
                                    //std::cout << "--no postop attr for " << it->getName() << std::endl;
                                    AddBytes(4, 0x05);    // 0x12c , no postop
                                }
                            }
                        }
                    }
                    break;
                case OpType::Softmax:
                    {

                        bSoftmax c = bSoftmax(&(*it));
                        next_offset += c.getSerializedSize() + 5*4;

                        // No more layers (last)
                        mv::DataModel dm(om);
                        mv::ControlModel cm(om);
                        Data::BufferIterator mem;
                        mv::Control::StageIterator stg = cm.getStage(0);
                        int finalstage = 0;
                        try{
                            auto t = it->getOutputTensor(0);
                            mem = dm.getBuffer("IntermediateMemory", stg, t);
                            if (mem == dm.bufferEnd("IntermediateMemory", stg)  ){
                                conv_pool_stage.next = 0;
                                finalstage = 1;
                            }
                        }catch(mv::ArgumentError){
                            printf("Warning: No Intermediary Buffers\n");
                            conv_pool_stage.next = 0;
                            finalstage = 1;
                        }

                        if (finalstage == 0){
                            conv_pool_stage.next = next_offset;
                        }
                        AddBytes(4, conv_pool_stage.next);
                        AddBytes(4, 0x03);                                // 0x60
                        AddBytes(4, conv_pool_stage.implementation);

                        c.writeStageInfo(&om, this);

                        AddBytes(4, 0x05);    // 0x12c , no preop
                        AddBytes(4, 0x05);    // 0x12c , no postop
                    }
                    break;
                case OpType::ReLU:
                    {

                        bRelu c = bRelu(&(*it));
                        next_offset += c.getSerializedSize() + 5*4;

                        // No more layers (last)
                        mv::DataModel dm(om);
                        mv::ControlModel cm(om);
                        Data::BufferIterator mem;
                        mv::Control::StageIterator stg = cm.getStage(0);
                        int finalstage = 0;
                        try{
                            auto t = it->getOutputTensor(0);
                            mem = dm.getBuffer("IntermediateMemory", stg, t);
                            if (mem == dm.bufferEnd("IntermediateMemory", stg)  ){
                                conv_pool_stage.next = 0;
                                finalstage = 1;
                            }
                        }catch(mv::ArgumentError){
                            printf("Warning: No Intermediary Buffers\n");
                            conv_pool_stage.next = 0;
                            finalstage = 1;
                        }
                        if (!finalstage){
                            conv_pool_stage.next = next_offset;
                        }
                        AddBytes(4, conv_pool_stage.next);
                        AddBytes(4, 0x06);
                        AddBytes(4, conv_pool_stage.implementation);

                        // Serialize for MyriadX H/W
                        c.writeStageInfo(&om, this);

                        AddBytes(4, 0x05);    // 0x12c , no preop
                        AddBytes(4, 0x05);    // 0x12c , no postop

                    }
                    break;
                case OpType::MaxPool2D:
                case OpType::AvgPool2D:
                    {

                        bPooling c = bPooling(&(*it));
                        next_offset += c.getSerializedSize() + 5*4;

                        // No more layers (last)
                        mv::DataModel dm(om);
                        mv::ControlModel cm(om);
                        Data::BufferIterator mem;
                        mv::Control::StageIterator stg = cm.getStage(0);
                        int finalstage = 0;
                        try{
                            auto t = it->getOutputTensor(0);
                            mem = dm.getBuffer("IntermediateMemory", stg, t);
                            if (mem == dm.bufferEnd("IntermediateMemory", stg)  ){
                                conv_pool_stage.next = 0;
                                finalstage = 1;
                            }
                        }catch(mv::ArgumentError){
                            printf("Warning: No Intermediary Buffers\n");
                            conv_pool_stage.next = 0;
                            finalstage = 1;
                        }
                        if(!finalstage){
                            conv_pool_stage.next = next_offset ;
                        }

                        AddBytes(4, conv_pool_stage.next);

                        if (it->getOpType() == OpType::MaxPool2D)
                        {
                            AddBytes(4, 0x01);
                        }
                        else if (it->getOpType() == OpType::AvgPool2D)
                        {
                            AddBytes(4, 0x02);
                        }
                        AddBytes(4, conv_pool_stage.implementation);

                        c.writeStageInfo(&om, this);

                        AddBytes(4, 0x05);    // 0x12c , no preop
                        AddBytes(4, 0x05);    // 0x12c , no postop

                    }
                    break;
                case OpType::Add:
                case OpType::Multiply:
                    {

                        bEltwise c = bEltwise(&(*it));
                        next_offset += c.getSerializedSize() + 5*4;

                        // No more layers (last)
                        mv::DataModel dm(om);
                        mv::ControlModel cm(om);
                        Data::BufferIterator mem;
                        mv::Control::StageIterator stg = cm.getStage(0);
                        int finalstage = 0;
                        try{
                            auto t = it->getOutputTensor(0);
                            mem = dm.getBuffer("IntermediateMemory", stg, t);
                            if (mem == dm.bufferEnd("IntermediateMemory", stg)  ){
                                finalstage = 1;
                                conv_pool_stage.next = 0;
                            }
                        }catch(mv::ArgumentError){
                            printf("Warning: No Intermediary Buffers\n");
                            finalstage = 1;
                            conv_pool_stage.next = 0;
                        }
                        if(!finalstage){
                            conv_pool_stage.next = next_offset ;
                        }
                        AddBytes(4, conv_pool_stage.next);


                        if (it->getOpType() == OpType::Add)
                        {
                            std::cout << "#### Regular EltAdd ####" << std::endl;

                            AddBytes(4, 0x0c);     // operation type element-wise Add
                        }
                        else if (it->getOpType() == OpType::Multiply)
                        {
                            AddBytes(4, 0x0d);     // operation type element-wise Multiply
                        }
                        AddBytes(4, conv_pool_stage.implementation);

                        c.writeStageInfo(&om, this);

                        AddBytes(4, 0x05);    // 0x12c , no preop
                        AddBytes(4, 0x05);    // 0x12c , no postop
                    }
                    break;
                case OpType::Scale:
                    {
                        std::cout << "We shouldn't have this case, because all scales should be absorbed?" << std::endl;

                        next_offset += 0x8c ;
                        AddBytes(4, conv_pool_stage.next);

                        if (it->getOpType() == OpType::Add)
                        {
                            std::cout << "#### THIS IS WHERE I AM ####" << std::endl;
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

                        try{
                            mem = dm.getBuffer("IntermediateMemory", stg, t);
                        }catch(mv::ArgumentError){
                            printf("Warning: No Intermediary Buffers\n");
                        }

                        if (mem == dm.bufferEnd("IntermediateMemory", stg)  ){
                            conv_pool_stage.next = 0;
                        }else{
                            conv_pool_stage.next = next_offset ;
                        }

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
        std::cout << stage_pad_size << " - " << buffer_section_offset << "? " << next_offset << std::endl;
        if (stage_pad_size > 0)
        {
            std::cout << "No:" << stage_pad_size << std::endl;
            AddBytes(stage_pad_size, 0x00000000);
        }

        //std::cout << "Finished writing stages section of blob" << std::endl;
    }

    void Blob_buffer::write_buffer_section(mv::ControlModel& cm)
    {

        std::cout << "--- Write Buffer ---" << std::endl;

        uint32_t buffer_header_pad_size = 3 ;
        uint32_t buffer_header_pad_val = 0x002a ;
        Float16Compressor cvtr ;

        mv::DataModel dm(cm);
        mv::Control::StageIterator stg = cm.getStage(0);

        // buffer section header
        AddBytes(4, (blob_stats.buffer_header_size + blob_stats.buffer_data_size));
        // AddBytes(4, (blob_stats.buffer_header_size + totalSize*2));

        for (unsigned i=0; i<buffer_header_pad_size; i++)
        {
            AddBytes(4, buffer_header_pad_val);
        }

        try{
            for(Data::BufferIterator bit = dm.bufferBegin("ConstantMemory", stg); bit != dm.bufferEnd("ConstantMemory", stg); ++bit){
                std::cout << "Buffer Size: " << bit->size << std::endl;
                for (int idx = 0; idx != (int)bit->size; idx++){
                    u_int16_t fp16_val = cvtr.compress((*bit->data).getData()[idx]) ;  // Convert to fp16.
                    // u_int16_t fp16_val = f32Tof16((*bit->data).getData()[idx]) ;  // Convert to fp16.
                    AddBytes(2, fp16_val) ;
                }

                int adjustment = 0;
                while((bit->size*2 + adjustment*2) % 64 != 0){
                    AddBytes(2, 0);
                    adjustment++;
                }
            }
        }catch(mv::ArgumentError){
            std::cout << "Warning: No Constant Memory Present." << std::endl;
        }
    }

    blob_summary Blob_buffer::getBlobSumm(){
        return this->blob_stats;
    }

    void Blob_buffer::write_relocation_section(mv::ControlModel& cm)
    {
        this->reloc_table.write(this);
    }
}
