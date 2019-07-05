#include "mcm/target/myriadx/nce1_utils.hpp"

mv::ConvolutionParameters mv::fillKernel2DOperationParameters(mv::Data::OpListIterator opIterator, bool add_padding)
{
    mv::ConvolutionParameters to_return;
    auto input_tensor = opIterator->getInputTensor(0);
    auto output_tensor = opIterator->getOutputTensor(0);

    auto input_dimensions = input_tensor->getShape();
    auto output_dimensions = output_tensor->getShape();

    if(opIterator->getOpType() == "Conv")
    {
        auto weigth_tensor = opIterator->getInputTensor(1);
        auto kernel_dimensions = weigth_tensor->getShape();
        to_return.kernel_width = kernel_dimensions[0];
        to_return.kernel_height = kernel_dimensions[1];
    }
    else if(opIterator->getOpType() == "AveragePool" || opIterator->getOpType() == "MaxPool")
    {
        auto kernel_dimensions = opIterator->get<std::array<short unsigned, 2>>("kSize");
        to_return.kernel_width = kernel_dimensions[0];
        to_return.kernel_height = kernel_dimensions[1];
    }

    to_return.input_width = input_dimensions[0];
    to_return.input_height = input_dimensions[1];
    to_return.input_channels = input_dimensions[2];
    to_return.output_width = output_dimensions[0];
    to_return.output_height = output_dimensions[1];
    to_return.output_channels = output_dimensions[2];

    if(add_padding)
    {
        std::vector<size_t> existing_output_tensor_paddings = output_tensor->get<std::vector<size_t>>("NCE1_Paddings");
        std::vector<size_t> existing_input_tensor_paddings = input_tensor->get<std::vector<size_t>>("NCE1_Paddings");

        to_return.input_width += existing_input_tensor_paddings[0];
        to_return.input_height += existing_input_tensor_paddings[1];
        to_return.input_channels += existing_input_tensor_paddings[2];
        to_return.output_width += existing_output_tensor_paddings[0];
        to_return.output_height += existing_output_tensor_paddings[1];
        to_return.output_channels += existing_output_tensor_paddings[2];
    }

    auto strides = opIterator->get<std::array<unsigned short, 2>>("stride");
    to_return.stride_vertical = strides[0];
    to_return.stride_horizontal = strides[1];

    auto paddings = opIterator->get<std::array<unsigned short, 4>>("padding");
    to_return.pad_x_up = paddings[2];
    to_return.pad_x_down = paddings[3];
    to_return.pad_y_left = paddings[0];
    to_return.pad_y_right = paddings[1];

    return to_return;
}

mv::MXDimensionsStrides mv::convertStrides(mv::Data::TensorIterator t, mv::ControlModel& cm, mv::DataModel& dm)
{
    MXDimensionsStrides toReturn;

    int fp16_size = 2;
    toReturn.dataType = 0;

    if (t == dm.tensorEnd())
        return toReturn;

    //Set MX Dimensions
    switch((int)t->getShape().ndims())
    {
        case 5:
        {
            // Hardware Weights
            toReturn.dimX = t->getShape()[0] * t->getShape()[4];
            toReturn.dimY = t->getShape()[1];
            toReturn.dimZ = t->getShape()[3] * t->getShape()[2];
        }
        break;
        case 4:
        {
            // Most Software Weights
            toReturn.dimZ = t->getShape()[3];
            toReturn.dimY = t->getShape()[2];
            toReturn.dimX = t->getShape()[0] * t->getShape()[1];
        }
        break;
        case 3:
        {
            // I/O
            toReturn.dimX = t->getShape()[0];
            toReturn.dimY = t->getShape()[1];
            toReturn.dimZ = t->getShape()[2];
        }
        break;
        case 2:
        {
            toReturn.dimX = 1;
            toReturn.dimY = t->getShape()[0];
            toReturn.dimZ = t->getShape()[1];
        }
        break;
        case 1:
        {
            toReturn.dimX = t->getShape()[0];
            toReturn.dimY = 1;
            toReturn.dimZ = 1;
        }
        break;
        default:
        {
            std::cout << "Serialization Error: Shape of Tensor not supported in graphFile serializer" << std::endl;
            assert(0);
        }

    }

    //2) Check allocators
    if (!dm.hasAllocator("ConstantMemory") || !dm.hasAllocator("IntermediateMemory"))
        throw RuntimeError(dm, "Required allocators missing");

    Data::BufferIterator mem;
    mv::Control::StageIterator stg = cm.getStage(0);

    unsigned D1_stride = 0, D1_block = 0;
    unsigned D2_stride = 0; //, D2_block = 0;
    int block = 0;

    bool is_tight = false;

    if (t->isPopulated())
    {
        std::cout << "Populated Tensor: " << t->getName() << t->getOrder().toString() << std::endl;
        mem = dm.getBuffer("ConstantMemory", stg, t);
        toReturn.location = BLOB_INTERNAL_LOCATION;
        toReturn.blocation = bLocation::Constant;
        is_tight = true;
        toReturn.pushToRelocationTable = true;
        toReturn.offset = mem->getOffset();

    }
    else//unpopulated
    {
        bool is_external = false;

        if (t->hasAttr("modelInput"))
        {
            if (t->get<bool>("modelInput"))
            {
                // Input tensor, non allocated in the blob
                toReturn.location = BLOB_INPUT_LOCATION;
                toReturn.offset = 0;
                toReturn.allocator_name = "ProgrammableInput";
            }
            else
                throw RuntimeError(*t, "Unallocated tensor marked as non input passed for serialization");
        }
        else if (t->hasAttr("modelOutput"))
        {
            if (t->get<bool>("modelOutput"))
            {
                // Output tensor, non allocated in the blob
                toReturn.location = BLOB_OUTPUT_LOCATION;
                toReturn.offset = 0;
                toReturn.allocator_name = "ProgrammableOutput";
            }
            else
                throw RuntimeError(*t, "Unallocated tensor marked as non output passed for serialization");
        }
        else
        {
            toReturn.location = BLOB_EXTERNAL_LOCATION;
            is_external = true;
            toReturn.allocator_name = "IntermediateMemory";
        }

        // Will throw IndexError on incorrect stage
        mem = dm.getBuffer(toReturn.allocator_name, stg, t);
        if (mem == dm.bufferEnd(toReturn.allocator_name, stg))
            throw RuntimeError(*t, "Unallocated tensor found during the serialization");

        unsigned leading_pad = 0;
        if (!mem->getStrides().empty())
        {
            bool Dim1_Stride_Set = false;

            // Start at 1 and go til -1 because the first and last strides are
            // leading and trailing "padding"
            for(std::size_t i = 1; i < mem->getStrides().size() - 2; i++)
            {
                unsigned blk_stride = (int)mem->getStrides()[i];
                block += (int)mem->getBlockSize();
                if (blk_stride != 0)
                {
                    if(Dim1_Stride_Set && blk_stride != D1_stride)
                    {
                        // 2nd dimension stride
                        D2_stride = blk_stride - D1_stride;  // wraparound
                        //D2_block = block;
                        break;  // no further striding support over 2D
                    }
                    else
                    {
                        D1_block = block;
                        block = 0;
                        D1_stride = blk_stride;
                        Dim1_Stride_Set = true;
                    }
                }
            }

            if(D1_stride == 0)
                is_tight = true;

            leading_pad = mem->getStrides()[0];

        }
        else
            is_tight = true;


        if(is_external)
        {
            toReturn.pushToRelocationTable = true;
            toReturn.blocation = bLocation::Variable;
            toReturn.offset = mem->getOffset() + leading_pad;
        }

    }

    int local_StrideX = 0;
    int local_StrideY = 0;
    int local_StrideZ = 0;

    if (is_tight)
    {
        // Tight or Empty Buffer. Either way no exterior striding
        std::cout << "Tight" << std::endl;
    }
    else
    {
        std::cout << "Not Tight" << std::endl;
        Order current_order = t->getOrder();
        if(current_order.isRowMajor())
        {
            if (D1_block == toReturn.dimZ * 2)
                local_StrideZ = D1_stride;
            else if (D1_block == toReturn.dimZ*toReturn.dimY * 2)
                local_StrideY = D1_stride;
            else if ( D1_block == toReturn.dimX*toReturn.dimY*toReturn.dimZ * 2)
                local_StrideX = D1_stride;
            else
                std::cout << "Serialization Error: Cannot figure out stride translation (ColumnMajor)" << std::endl;
        }

        else if(current_order.isColMajor())
        {
            // This is horrible because values in allocator are 5D
            if (D1_stride == toReturn.dimZ* 2 * 8)
                local_StrideY = D1_stride - 16;

            else if (D1_stride == toReturn.dimY*toReturn.dimZ* 2  * 8)
                local_StrideZ = D1_stride - 16;

            else if ( D1_stride == toReturn.dimX*toReturn.dimY*toReturn.dimZ * 2 * 8)
                local_StrideX = D1_stride - 16;

            else
                std::cout << "Serialization Error: Cannot figure out stride translation (RowMajor)" << D1_block << std::endl;
        }

        else if(current_order.isRowMajorPlanar())
        {
            if (D1_block == toReturn.dimZ * 2)
                local_StrideZ = D1_stride;
            else if (D1_block == toReturn.dimX*toReturn.dimY* 2)
                local_StrideX = D1_stride;
            else if ( D1_block == toReturn.dimX*toReturn.dimY*toReturn.dimZ * 2)
                local_StrideY = D1_stride;
            else
                std::cout << "Serialization Error: Cannot figure out stride translation (RowMajorPlanar)" << std::endl;
        }


        else if(current_order.isColMajorPlanar())
        {

            if (D1_block == toReturn.dimY* 2)
                local_StrideY = D1_stride;
            else if (D1_block == toReturn.dimX*toReturn.dimY* 2)
                local_StrideX = D1_stride;
            else if ( D1_block == toReturn.dimX*toReturn.dimY*toReturn.dimZ * 2)
                local_StrideZ = D1_stride;
            else
                std::cout << "Serialization Error: Cannot figure out stride translation (ColumnMajorPlanar)" << std::endl;
        }


        else if(current_order.isRowInterleaved())
        {
            if (D1_block == toReturn.dimX * 2)
                local_StrideX = D1_stride;
            else if (D1_block == toReturn.dimZ*toReturn.dimX * 2)
                local_StrideZ = D1_stride;

            else if ( D1_block == toReturn.dimX*toReturn.dimY*toReturn.dimZ * 2 )
                local_StrideY = D1_stride;
            else
                std::cout << "Serialization Error: Cannot figure out stride translation (RowInterleaved) Block Size: " << D1_block << std::endl;
            if(D2_stride != 0)
                local_StrideZ = D2_stride;
        }
    }

    Order current_order = t->getOrder();

    if(current_order.isRowMajorPlanar())
    {
        if(t->getShape().ndims() == 3)
        {
            // UPA Shave
            toReturn.order = 0;
            // ROW MAJOR (CHANNEL MINOR)
            // I.E: Y, X, Z
            toReturn.strideZ = fp16_size;
            toReturn.strideX = (toReturn.dimZ * toReturn.strideZ) + local_StrideZ;
            toReturn.strideY = (toReturn.dimX * toReturn.strideX) + local_StrideX;
        }
        else
        {
            if(t->getShape().ndims() > 3)
            {
                // Software weights follow a different paradigm in c++ and python/mvtensor, causing this case.
                // MvTensor actually uses ZYX rather than ZXY here. (confusion caused by multidimensionality)
                toReturn.order = 3;
                toReturn.strideZ = fp16_size;
                toReturn.strideY = (toReturn.dimZ * toReturn.strideZ) + local_StrideZ;
                toReturn.strideX = (toReturn.dimY * toReturn.strideY) + local_StrideY;
            }
            else
            {
                // Software weights follow a different paradigm in c++ and python/mvtensor, causing this case.
                // MvTensor actually uses ZYX rather than ZXY here. (confusion caused by multidimensionality)
                toReturn.order = 1;
                toReturn.strideX = fp16_size;
                toReturn.strideY = (toReturn.dimX * toReturn.strideX) + local_StrideX;
                toReturn.strideZ = (toReturn.dimY * toReturn.strideY) + local_StrideY;
            }
        }
    }

    else if(current_order.isRowMajor())
    {
        // NCE1 - Option 1
        // COLUMN MAJOR(NCE1 Planar)
        // I.E: X, Y, Z
        if(t->getShape().ndims() > 3)
        {
            toReturn.order = 1;    // THIS ENUM IS WRONG
            toReturn.strideX = fp16_size;
            toReturn.strideY = (toReturn.dimX * toReturn.strideX) + local_StrideX;
            toReturn.strideZ = (toReturn.dimY * toReturn.strideY) + local_StrideY;
        }
        else
        {
            // Misleading - weights
            toReturn.order = 3;
            toReturn.strideZ = fp16_size;
            toReturn.strideY = (toReturn.dimZ * toReturn.strideZ) + local_StrideZ;
            toReturn.strideX = (toReturn.dimY * toReturn.strideY) + local_StrideY;
        }
    }
    else if(current_order.isColMajor())
    {
        // Misleading - weights
        toReturn.order = 3;
        toReturn.strideZ = fp16_size;
        toReturn.strideY = (toReturn.dimZ * toReturn.strideZ) + local_StrideZ;
        toReturn.strideX = (toReturn.dimY * toReturn.strideY) + local_StrideY;
    }
    else if(current_order.isColMajorPlanar())
    {
        toReturn.order = 1;
        toReturn.strideY = fp16_size;
        toReturn.strideX = (toReturn.dimY * toReturn.strideY) + local_StrideY;
        toReturn.strideZ = (toReturn.dimX * toReturn.strideX) + local_StrideX;
        // toReturn.strideX = fp16_size;
        // toReturn.strideY = (toReturn.dimX + local_StrideX)*toReturn.strideX;
        // toReturn.strideZ = (toReturn.dimY + local_StrideY)*toReturn.strideY;
    }
    else if(current_order.isRowInterleaved())
    {
        toReturn.order = 2;
        toReturn.strideX = fp16_size;
        toReturn.strideZ = (toReturn.dimX * toReturn.strideX) + local_StrideX;
        toReturn.strideY = (toReturn.dimZ * toReturn.strideZ) + local_StrideZ;
    }

    // std::cout << "Order: " << (*t)->getOrder().toString() << std::endl;
    std::cout << "X: Dim:" << toReturn.dimX << ", Stride: " << toReturn.strideX << "(local: " << local_StrideX << ")" << std::endl;
    std::cout << "Y: Dim:" << toReturn.dimY << ", Stride: " << toReturn.strideY << "(local: " << local_StrideY << ")" << std::endl;
    std::cout << "Z: Dim:" << toReturn.dimZ << ", Stride: " << toReturn.strideZ << "(local: " << local_StrideZ << ")" << std::endl;
    // std::cout << "Block size: " << D1_block << "|" << D2_block << std::endl;
    // std::cout << "Block stride: " << D1_stride << "|" << D2_stride << std::endl;
    // std::cout << "Strides:" << D1_stride << "," << D2_stride << std::endl;
    // std::cout << "Blocks:" << D1_block << "," << D2_block << std::endl;

    return toReturn;
}
