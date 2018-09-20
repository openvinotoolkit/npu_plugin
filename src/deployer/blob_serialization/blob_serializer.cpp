#include "include/mcm/deployer/serializer.hpp"

#include <climits>
#include <stdio.h>

namespace mv
{

    int Blob_buffer::get_blob_enum(mv::OpType o, bool NCE1) {
        /***
         *  Mapping of C++ OpTypes to Blob Enumerations
         ***/
        switch ((unsigned short)o) {
            case OpType::Conv2D:
                {
                    if (NCE1) {
                        return 33;
                    } else {
                        return 0;
                    }
                }
            case OpType::MaxPool2D:
                return 1;
            case OpType::AvgPool2D:
                return 2;
            case OpType::Softmax:
                return 3;
            case OpType::FullyConnected:
                return 4;
            case OpType::Input:
            case OpType::Output:
                return 5;   // NoOp
            case OpType::ReLU:
                return 6;
            case OpType::Add:
                return 12;
            case OpType::Multiply:
                return 13;
            case OpType::Scale:
                return 15;
            case OpType::Conversion:
                return 37;
            default:
                {
                    std::cout << "Serialization Error: No Blob Enum Defined for layer" << o.toString() << std::endl;
                    assert(0);
                }
        }
    }

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

        mv::DataModel dm(cm);

        for (mv::Control::OpDFSIterator it = cm.getFirst(); it != cm.opEnd(); ++it)
        {
            switch((unsigned short)it->getOpType()){
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
                                mx_valid = it->get<int>("NCE1_Compatible");
                            }

                            if(mx_valid){

                                int descriptors = 1;
                                if (! it->hasAttr("NCE1_DescriptorSplits"))
                                {
                                    printf("Warning: attribute NCE1_DescriptorSplits not present. Defaulting to 1.\n");
                                }
                                else
                                {
                                    descriptors = it->get<int>("NCE1_DescriptorSplits");
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
                            uint32_t buffer_bias_values_len = dm.findTensor(it->get<std::string>("bias"))->getData().size() ;
                            blob_stats.bias_region_size += buffer_bias_values_len*blob_stats.weights_number_size;
                            blob_stats.data_buffer_count++ ;
                        }

                        blob_stats.stage_count++ ;
                        if (it->hasAttr("postOpType"))
                        {
                            if (it->get<mv::OpType>("postOpType") == mv::OpType::ReLU)
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
                        blob_stats.stage_section_size += bRelu::getSerializedSize()+5*4 ;
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
                        blob_stats.stage_section_size += bCompatibility::getSerializedSize()+5*4 ;
                    }
                    break;
                case OpType::Input:
                case OpType::Output:
                    {}
                    break;

                default:
                    std::cout << "Serialization Warning : The layer has not been used in calculation:" << it->getOpType().toString() << std::endl;
                    break;
            }
        }

        blob_stats.output_size = cm.getLast()->getInputTensor(0)->getShape().totalSize();
        blob_stats.stage_section_size = align(blob_stats.stage_section_size, 16) ;

        // Calculate Buffer Size
        mv::Control::StageIterator stg = cm.getStage(0);

        unsigned int totalSize = 0;

        try
        {
            for (Data::BufferIterator bit = dm.bufferBegin("ConstantMemory", stg); bit != dm.bufferEnd("ConstantMemory", stg); ++bit)
            {
                totalSize += bit->getSize();
                int adjustment = 0;
                while((bit->getSize()*2 + adjustment*2) % 64 != 0)
                {
                    adjustment++;
                }
                totalSize += adjustment;
            }
        }
        catch(mv::ArgumentError)
        {
            std::cout << "Warning: No Constant Memory Present." << std::endl;
        }

        blob_stats.buffer_data_size = totalSize*2 ;

        blob_stats.relocation_section_size = 20 + 8*blob_stats.data_buffer_count + 16*(blob_stats.stage_count-2) + (8*blob_stats.elt_count) + additional_buf*8;

        blob_stats.blob_file_size = headers_data_size+blob_stats.header_pad_size+blob_stats.stage_section_size+blob_stats.buffer_header_size+blob_stats.buffer_data_size+blob_stats.relocation_section_size ;
    }

    void Blob_buffer::write_elf_header(){
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

    void Blob_buffer::write_mv_header(){

        uint32_t mv_magic_number = BLOB_MAGIC_NUMBER;
        uint32_t mv_version_major = BLOB_VERSION_MAJOR;
        uint32_t mv_version_minor = BLOB_VERSION_MINOR;
        uint32_t mv_num_shaves = 1;

        uint32_t mv_stage_section_offset = blob_stats.elf_header_size +
            blob_stats.mv_header_size + blob_stats.header_pad_size;
        uint32_t mv_buffer_section_offset = mv_stage_section_offset +
            blob_stats.stage_section_size;
        uint32_t mv_relocation_offset = mv_buffer_section_offset +
            blob_stats.buffer_header_size + blob_stats.buffer_data_size;
        uint32_t mv_permutation_enabled = 0x0000;

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
        mv::OpModel om(cm);
        mv::DataModel dm(cm);

        // Write each stage as we encounter it.
        for (mv::Control::OpDFSIterator it = cm.getFirst(); it != cm.opEnd(); ++it)
        {

            auto ltype = it->getOpType();
            switch((unsigned short)ltype){
                case OpType::Input:
                    {
                        AddBytes(4, 0x20);     // include input NoOp stage for compatibility with python compiler
                        AddBytes(4, get_blob_enum(ltype));
                        AddBytes(4, BLOB_DEFAULT_IMPLEMENTATION);
                        AddBytes(4, 0x05);
                        AddBytes(4, 0x05);
                    }
                    break;

                case OpType::Conv2D:
                    {
                        int mx_valid = 0;
                        if (! it->hasAttr("NCE1_Compatible"))
                        {
                            printf("Warning: attribute NCE1_Compatible not present. Assuming False.\n");
                        }
                        else
                        {
                            mx_valid = it->get<int>("NCE1_Compatible");
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
                                descriptors = it->get<int>("NCE1_DescriptorSplits");
                            }
                            point0 += (11*4) ; // Header of Descriptors
                            point0 += (descriptors*32*4) ; // Descriptor
                            point0 += (5*10*4) ; // Input, Bias, Taps, Output, Scale

                            point0 += (2*4) ; // PreOp PostOp TODO: Move OUT.
                            point0 += (3*4) ; // nextsatge, etc MOVE OUT.

                            next_offset += point0 ;


                            // No more layers (last)
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
                            }catch(mv::IndexError){
                                printf("Warning: No Intermediary Buffers\n");
                                conv_pool_stage.next = 0;
                                finalstage = 1;
                            }
                            if(!finalstage){
                                conv_pool_stage.next = next_offset;
                            }

                            AddBytes(4, conv_pool_stage.next);
                            AddBytes(4, get_blob_enum(ltype, true));
                            AddBytes(4, BLOB_DEFAULT_IMPLEMENTATION);

                            // Serialize for MyriadX H/W
                            bConv2D c = bConv2D(&(*it));
                            c.writeStageInfo(&om, this);

                            AddBytes(4, 0x05);    // 0x12c , no preop
                            AddBytes(4, 0x05);    // 0x12c , no postop
                        }
                        else
                        {
                            // Serialize for S/W
                            int point0 = 0;
                            point0 += (8*4) ; // Fields
                            point0 += (4*10*4) ; // Input, Bias, Taps, Output, Scale
                            point0 += (3*4) ; // nextsatge, etc MOVE OUT.


                            if (it->hasAttr("postOpType"))
                            {
                                if (it->get<mv::OpType>("postOpType") == mv::OpType::ReLU)
                                {
                                    point0 += (5*4) ;
                                }
                                else
                                {
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
                            }catch(mv::IndexError){
                                printf("Warning: No Intermediary Buffers\n");
                                conv_pool_stage.next = 0;
                                finalstage = 1;
                            }
                            if(!finalstage){
                                conv_pool_stage.next = next_offset;
                            }

                            AddBytes(4, conv_pool_stage.next);
                            AddBytes(4, get_blob_enum(ltype));                                // 0x60
                            AddBytes(4, BLOB_DEFAULT_IMPLEMENTATION);

                            // Serialize for MyriadX H/W
                            bConv2D c = bConv2D(&(*it));
                            c.writeStageInfo(&om, this);

                            AddBytes(4, conv_pool_stage.preop_type);
                            if (it->hasAttr("postOpType"))
                            {

                                if (it->get<mv::OpType>("postOpType") == mv::OpType::ReLU)
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
                        // Currently not triggered - converted to Conv.
                        int point0 = 0;
                        point0 += 4*10 ; // Input, Output
                        point0 += 2 ; // PreOp PostOp TODO: Move OUT.
                        point0 += 3 ; // nextstage, id, imp
                        next_offset += point0*4 ;

                        // No more layers (last)
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
                        }catch(mv::IndexError){
                            printf("Warning: No Intermediary Buffers\n");
                            conv_pool_stage.next = 0;
                            finalstage = 1;
                        }
                        if (!finalstage){
                            conv_pool_stage.next = next_offset;
                        }
                        AddBytes(4, conv_pool_stage.next);
                        AddBytes(4, get_blob_enum(ltype));                                // 0x60
                        AddBytes(4, BLOB_DEFAULT_IMPLEMENTATION);

                        // Serialize for MyriadX H/W
                        bInnerProduct c = bInnerProduct(&(*it));
                        c.writeStageInfo(&om, this);

                        AddBytes(4, 0x05);    // 0x12c , no preop
                        AddBytes(4, 0x05);    // 0x12c , no postop

                    }
                    break;
                case OpType::Softmax:
                    {

                        bSoftmax c = bSoftmax(&(*it));
                        next_offset += c.getSerializedSize() + 5*4;

                        // No more layers (last)
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
                        }catch(mv::IndexError){
                            printf("Warning: No Intermediary Buffers\n");
                            conv_pool_stage.next = 0;
                            finalstage = 1;
                        }

                        if (finalstage == 0){
                            conv_pool_stage.next = next_offset;
                        }
                        AddBytes(4, conv_pool_stage.next);
                        AddBytes(4, get_blob_enum(ltype));                                // 0x60
                        AddBytes(4, BLOB_DEFAULT_IMPLEMENTATION);

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
                        }catch(mv::IndexError){
                            printf("Warning: No Intermediary Buffers\n");
                            conv_pool_stage.next = 0;
                            finalstage = 1;
                        }
                        if (!finalstage){
                            conv_pool_stage.next = next_offset;
                        }
                        AddBytes(4, conv_pool_stage.next);
                        AddBytes(4, get_blob_enum(ltype));
                        AddBytes(4, BLOB_DEFAULT_IMPLEMENTATION);

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
                        }catch(mv::IndexError){
                            printf("Warning: No Intermediary Buffers\n");
                            conv_pool_stage.next = 0;
                            finalstage = 1;
                        }
                        if(!finalstage){
                            conv_pool_stage.next = next_offset ;
                        }

                        AddBytes(4, conv_pool_stage.next);
                        AddBytes(4, get_blob_enum(ltype));
                        AddBytes(4, BLOB_DEFAULT_IMPLEMENTATION);

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
                        }catch(mv::IndexError){
                            printf("Serializer Warning: No Intermediary Buffers\n");
                            finalstage = 1;
                            conv_pool_stage.next = 0;
                        }
                        if(!finalstage){
                            conv_pool_stage.next = next_offset ;
                        }
                        AddBytes(4, conv_pool_stage.next);
                        AddBytes(4, get_blob_enum(ltype));
                        AddBytes(4, BLOB_DEFAULT_IMPLEMENTATION);

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

                        AddBytes(4, get_blob_enum(ltype));     // operation type element-wise Add

                        next_offset += 0x28 ;

                        AddBytes(4, BLOB_DEFAULT_IMPLEMENTATION);

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

                        bCompatibility c = bCompatibility(&(*it));
                        next_offset += c.getSerializedSize() + 5*4 ;

                        Data::BufferIterator mem;
                        mv::Control::StageIterator stg = cm.getStage(0);
                        auto t = it->getOutputTensor(0);

                        try{
                            mem = dm.getBuffer("IntermediateMemory", stg, t);
                        }catch(mv::IndexError){
                            printf("Warning: No Intermediary Buffers\n");
                        }

                        if (mem == dm.bufferEnd("IntermediateMemory", stg)  ){
                            conv_pool_stage.next = 0;
                        }else{
                            conv_pool_stage.next = next_offset ;
                        }

                        AddBytes(4, conv_pool_stage.next);
                        AddBytes(4, get_blob_enum(ltype));                                // 0x60
                        AddBytes(4, BLOB_DEFAULT_IMPLEMENTATION);

                        // Serialize for MyriadX H/W
                        c.writeStageInfo(&om, this);

                        AddBytes(4, 0x05);    // 0x12c , no preop
                        AddBytes(4, 0x05);    // 0x12c , no postop

                    }
                    break;

                default:
                    break;
                    //std::cout << "Serialization Error: No Available Write Methods for layer:" << Printable::toString(it->getOpType()) << std::endl;
                    //assert(0);
            }
        }

        uint32_t buffer_section_offset = align(next_offset,0x10) ;
        uint32_t stage_pad_size = buffer_section_offset - next_offset  ;
        if (stage_pad_size > 0)
        {
            AddBytes(stage_pad_size, 0x00000000);
        }

    }

    void Blob_buffer::write_buffer_section(mv::ControlModel& cm)
    {

        std::cout << "--- Write Buffer ---" << std::endl;

        uint32_t buffer_header_pad_size = 3 ;
        uint32_t buffer_header_pad_val = 0x002a ;
        mv_num_convert cvtr ;

        mv::DataModel dm(cm);
        mv::Control::StageIterator stg = cm.getStage(0);

        // buffer section header
        AddBytes(4, (blob_stats.buffer_header_size + blob_stats.buffer_data_size));

        for (unsigned i=0; i<buffer_header_pad_size; i++)
        {
            AddBytes(4, buffer_header_pad_val);
        }

        try{
            std::vector<mv::MemoryAllocator::MemoryBuffer> buffers_out_of_order, buffers_in_order;

            // TODO: Needs an iterator that goes through items in ascending offset order.
            for(Data::BufferIterator bbit = dm.bufferBegin("ConstantMemory", stg); bbit != dm.bufferEnd("ConstantMemory", stg); ++bbit){
                buffers_out_of_order.push_back(*bbit);
            }

            int tsize = buffers_out_of_order.size();
            for(int i = 0; i != tsize; i++)
            {
                mv::MemoryAllocator::MemoryBuffer smallest;
                unsigned long long smallest_val = ULLONG_MAX;   // If this is ever hit we have far, far bigger problems :)
                for (mv::MemoryAllocator::MemoryBuffer m : buffers_out_of_order)
                {
                    if(smallest_val > m.getOffset())
                    {
                        smallest_val = m.getOffset();
                        smallest = m;
                    }
                }
                if (smallest_val == ULLONG_MAX)
                {
                    break;
                }
                buffers_in_order.push_back(smallest);
                int pos = find(buffers_out_of_order.begin(), buffers_out_of_order.end(), smallest) - buffers_out_of_order.begin();

                buffers_out_of_order.erase(buffers_out_of_order.begin() + pos);

            }

            unsigned int running_total = 0;
            for(auto bit : buffers_in_order)
            {
                running_total += bit.getSize()*2;

                for (int idx = 0; idx != (int)bit.getSize(); idx++){
                    u_int16_t fp16_val = cvtr.fp32_to_fp16(static_cast<float>(bit.getData()->getData()[idx])) ;  // Convert to fp16.
                    AddBytes(2, fp16_val) ;
                }

                // TODO: To be removed when allocater takes care of this.
                int adjustment = 0;
                while((bit.getSize()*2 + adjustment*2) % 64 != 0)
                {
                    AddBytes(2, 0);
                    adjustment++;
                    running_total += 2;
                }
            }
        }catch(mv::ArgumentError){
            std::cout << "Warning: No Constant Memory Present." << std::endl;
        }
    }

    blob_summary Blob_buffer::getBlobSumm(){
        return this->blob_stats;
    }

    void Blob_buffer::write_relocation_section(mv::ControlModel&)
    {
        this->reloc_table.write(this);
    }
}
