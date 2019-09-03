#include "include/mcm/deployer/serializer.hpp"
#include "include/mcm/utils/custom_math.hpp"

#include <climits>
#include <stdio.h>

namespace mv
{
    uint32_t Blob_buffer::calc(mv::ControlModel& cm, mv::TargetDescriptor& td)
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
        blob_stats.stage_count = 0 ;     // start count including NoOp stage
        blob_stats.data_buffer_count = 0 ;
        blob_stats.elt_count = 0 ;
        blob_stats.stage_section_size = 4*3 ;    // start count including 12 byte header and NoOp stage
        blob_stats.weights_region_size = 0 ;
        blob_stats.bias_region_size = 0 ;

        mv::DataModel dm(cm);

        unsigned tensor_count = 0;

        for (mv::Control::OpDFSIterator it = cm.getFirst(); it != cm.opEnd(); ++it)
        {

            mv::Element e("serial_viewer");
            if (it->getOpType() == "Input")
            {
                blob_stats.stage_section_size += 4*5;
                blob_stats.stage_count++;
                continue;
            }
            else if(it->getOpType() == "Output"
                || it->getOpType() == "Constant"
                || it->getOpType() == "Concat")
                continue;

            else if (it->hasAttr("NCE1_Compatible") && it->get<int>("NCE1_Compatible"))
                e = td.getSerialDefinition(it->getOpType(), "NCE1");
            else
                e = td.getSerialDefinition(it->getOpType(), "MvTensor");
            std::vector<std::string> serial_instructions = e.get<std::vector<std::string>>("serial_view");

            // Calculate Offset to Next Pointer
            for(auto s = serial_instructions.begin(); s != serial_instructions.end(); ++s){
                std::string instruction = s->substr(0, s->find(':'));
                std::string name = s->substr(s->find(':')+1, s->size());
                if(instruction == "Attr")
                {
                    auto attr = it->get(name);
                    auto b = attr.toBinary();
                    blob_stats.stage_section_size += b.size();
                }
                else if(instruction == "Tensor")
                {
                    blob_stats.stage_section_size += 4*10;

                    // Only want to insert entries into reloc table that exist
                    std::string inOrOut = name.substr(0, name.find(':'));
                    std::string index = name.substr(name.find(':')+1, name.size());
                    mv::Data::TensorIterator retrievedT;
                    if(inOrOut == "0")
                    {
                        unsigned idx = stoi(index);
                        try
                        {
                            retrievedT = it->getInputTensor(idx);
                        }
                        catch (...)
                        {
                            retrievedT = dm.tensorEnd();
                        }
                    }
                    else
                    {
                        unsigned idx = stoi(index);
                        retrievedT = it->getOutputTensor(idx);
                    }
                    if(retrievedT != dm.tensorEnd())
                        tensor_count++;
                }
            }
            blob_stats.stage_section_size += 4*5;   //3 meta fields, 2 post pre op

            blob_stats.stage_count++;
        }

        std::cout << "stage_section_size" << blob_stats.stage_section_size << std::endl;

        blob_stats.output_size = cm.getLast()->getInputTensor(0)->getShape().totalSize();
        blob_stats.stage_section_size = align(blob_stats.stage_section_size, 16) ;

        std::cout << "stage_section_size" << blob_stats.stage_section_size << std::endl;

        // Calculate Buffer Size
        mv::Control::StageIterator stg = cm.getStage(0);

        unsigned int totalSize = 0;

        if (dm.iterable("ConstantMemory", stg))
        {
            for (Data::BufferIterator bit = dm.bufferBegin("ConstantMemory", stg); bit != dm.bufferEnd("ConstantMemory", stg); ++bit)
            {
                totalSize += bit->getSize() ;
                totalSize += bit->getPostAlign();
            }
        }

        blob_stats.buffer_data_size = totalSize;

        std::cout << tensor_count << std::endl;

        tensor_count -= 2;  // No output or input relocation entries

        blob_stats.relocation_section_size =
            20 +
            8 * tensor_count;

        blob_stats.blob_file_size =
            headers_data_size +
            blob_stats.header_pad_size +
            blob_stats.stage_section_size +
            blob_stats.buffer_header_size +
            blob_stats.buffer_data_size +
            blob_stats.relocation_section_size;

        return blob_stats.blob_file_size ;
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

    void Blob_buffer::write_ops(mv::ComputationModel& model, mv::TargetDescriptor& td){
        mv::OpModel om(model);
        mv::DataModel dm(model);
        mv::ControlModel cm(model);

        unsigned next_offset = 4*3;     // Who knows?

        for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
        {
            std::cout << "Writing Serial Fields for Op{" << opIt->getOpType() << "}" <<std::endl;

            mv::Element e("serial_viewer");
            if (opIt->getOpType() == "Input")
            {
                AddBytes(4, 0x20);
                AddBytes(4, 5);
                AddBytes(4, BLOB_DEFAULT_IMPLEMENTATION);
                AddBytes(4, 5);
                AddBytes(4, 5);
                next_offset += 4*5;

                continue;
            }
            else if(opIt->getOpType() == "Output"
                || opIt->getOpType() == "Constant"
                || opIt->getOpType() == "Concat")
                continue;
            else if (opIt->hasAttr("NCE1_Compatible") && opIt->get<int>("NCE1_Compatible"))
                e = td.getSerialDefinition(opIt->getOpType(), "NCE1");
            else
                e = td.getSerialDefinition(opIt->getOpType(), "MvTensor");

            std::vector<std::string> serial_instructions = e.get<std::vector<std::string>>("serial_view");

            // Calculate Offset to Next Pointer
            for(auto s = serial_instructions.begin(); s != serial_instructions.end(); ++s)
            {
                std::string instruction = s->substr(0, s->find(':'));
                std::string name = s->substr(s->find(':')+1, s->size());
                if(instruction == "Attr")
                {
                    auto attr = opIt->get(name);
                    auto b = attr.toBinary();
                    next_offset += b.size();
                }
                else if(instruction == "Tensor")
                    next_offset += 4*10;

            }
            next_offset += 3*4 + 2*4;   // First few fields, Post/Pre Ops


            unsigned offset = next_offset;

            if(opIt->getOutputTensor(0) == om.getOutput()->getInputTensor(0))
                offset = 0;// Final Layer. Unreliable Method to obtain...
            else
                offset = next_offset;

            // Some Construction and other Fields
            AddBytes(4, offset);
            AddBytes(4, opIt->get<unsigned>("SerialID"));   // TODO: Enum registers
            AddBytes(4, BLOB_DEFAULT_IMPLEMENTATION);

            for(auto s = serial_instructions.begin(); s != serial_instructions.end(); ++s)
            {
                std::string instruction = s->substr(0, s->find(':'));
                std::string name = s->substr(s->find(':')+1, s->size());
                if(instruction == "Attr")
                {
                    auto attr = opIt->get(name);
                    auto b = attr.toBinary();
                    for( uint8_t byte : b)
                        AddBytes(1, byte);

                }
                else if(instruction == "Tensor")
                {
                    std::string inOrOut = name.substr(0, name.find(':'));
                    std::string index = name.substr(name.find(':')+1, name.size());
                    mv::Data::TensorIterator retrievedT;
                    std::cout << inOrOut << ":" << index << std::endl;
                    if(inOrOut == "0")
                    {
                        unsigned idx = stoi(index);
                        try
                        {
                            retrievedT = opIt->getInputTensor(idx);
                        }
                        catch(...)
                        {
                            retrievedT = dm.tensorEnd();
                        }
                    }
                    else
                    {
                        unsigned idx = stoi(index);
                        retrievedT = opIt->getOutputTensor(idx);
                    }
                    if(retrievedT == dm.tensorEnd())
                        std::cout << "Retrieved NULL: " << ": " << std::endl;
                    else
                        std::cout << "Retrieved Tensor: " << ": " << retrievedT->getName() << std::endl;

                    Blob_Tensor bt = Blob_Tensor(dm, cm, this->reloc_table, retrievedT);
                    bt.write(this);
                }
                else
                {
                    // throw mv::AttributeError(instruction, "Invalid Serialization Instruction");
                }
            }

            AddBytes(4, 5);     // Post/Pre Op (Deprecated)
            AddBytes(4, 5);     // Post/Pre Op (Deprecated)

        }

        uint32_t buffer_section_offset = align(next_offset, 16);
        uint32_t stage_pad_size = buffer_section_offset - next_offset;
        if (stage_pad_size > 0)
            AddBytes(stage_pad_size, 0x00000000);
    }

    void Blob_buffer::write_buffer_section(mv::ControlModel& cm)
    {

        std::cout << "--- Write Buffer ---" << std::endl;

        uint32_t buffer_header_pad_size = 3 ;
        uint32_t buffer_header_pad_val = 0x002a ;

        mv::DataModel dm(cm);
        mv::Control::StageIterator stg = cm.getStage(0);

        // buffer section header
        AddBytes(4, (blob_stats.buffer_header_size + blob_stats.buffer_data_size));

        for (unsigned i=0; i<buffer_header_pad_size; i++)
            AddBytes(4, buffer_header_pad_val);


        if (dm.iterable("ConstantMemory", stg))
        {
            for(auto bit = dm.bufferBegin("ConstantMemory", stg); bit != dm.bufferEnd("ConstantMemory", stg); ++bit)
            {
                bool tight = true;
                for ( auto s : bit->getStrides() )
                    if (s != 0)
                        tight = false;

                // Push tensor's data
                if (tight)
                {
                    //auto data = bit->getData()->getData();
                    for (std::size_t idx = 0; idx != bit->getData()->getShape().totalSize(); idx++)
                    {
                        uint16_t fp16_val = mv::fp32_to_fp16(static_cast<float>(bit->getData()->at(idx)));  // Convert to fp16.
                        AddBytes(2, fp16_val);
                    }
                }
                else
                {
                    uint16_t fp16_val;
                    for (std::size_t block_idx = 0; block_idx != bit->getBlockNum(); block_idx++)
                    {
                        // TODO: lhs stride
                        for (std::size_t elem_idx = 0; elem_idx != bit->getStrides()[block_idx] / 2; elem_idx++)    // TODO: not only FP16
                        {
                            //std::cout << "x" ;
                            fp16_val = mv::fp32_to_fp16(static_cast<float>(0));  // Convert to fp16.
                            AddBytes(2, fp16_val);
                        }

                        //auto data = bit->getData()->getData();
                        for (std::size_t elem_idx = 0; elem_idx != (bit->getBlockSize() / 2); elem_idx++)    // TODO: not only FP16
                        {
                            //std::cout << "o" ;
                            uint16_t idx = ((block_idx*bit->getBlockSize())/2) + elem_idx;
                            fp16_val = mv::fp32_to_fp16(static_cast<float>(bit->getData()->at(idx)));  // Convert to fp16.
                            AddBytes(2, fp16_val);
                        }
                        //std::cout << std::endl;
                    }
                    for (std::size_t elem_idx = 0; elem_idx < bit->getStrides()[bit->getBlockNum()] / 2; elem_idx++)    // TODO: not only FP16
                    {
                        fp16_val = mv::fp32_to_fp16(static_cast<float>(0));  // Convert to fp16.
                        AddBytes(2, fp16_val);
                    }
               }

                // Push alignment
                for (std::size_t i = 0; i < bit->getPostAlign(); ++i)
                    AddBytes(1, 0);
            }
        }
    }

    blob_summary Blob_buffer::getBlobSumm()
    {
        return this->blob_stats;
    }

    void Blob_buffer::write_relocation_section(mv::ControlModel&)
    {
        this->reloc_table.write(this);
    }
}
