#include <stdio.h>
#include "include/mcm/deployer/blob_serialization/bTensor.hpp"
#include <string.h>

#define BLOB_INPUT_LOCATION 1
#define BLOB_OUTPUT_LOCATION 2
#define BLOB_INTERNAL_LOCATION 3
#define BLOB_EXTERNAL_LOCATION 4

namespace mv
{
    void Blob_Tensor::write(WBuffer* b)
    {
        b->AddBytes(4, this->dimX);
        b->AddBytes(4, this->dimY);
        b->AddBytes(4, this->dimZ);
        b->AddBytes(4, this->strideX);
        b->AddBytes(4, this->strideY);
        b->AddBytes(4, this->strideZ);
        b->AddBytes(4, this->offset);
        b->AddBytes(4, this->location);
        b->AddBytes(4, this->dataType);
        b->AddBytes(4, this->order);
    }

    Blob_Tensor::Blob_Tensor(int x, int y, int z,
        int sx, int sy, int sz,
        int offsetParam, int locationParam,
        int dtype, int orderParam)
        : dimX(x),
          dimY(y),
          dimZ(z),
          strideX(sx),
          strideY(sy),
          strideZ(sz),
          offset(offsetParam),
          location(locationParam),
          dataType(dtype),
          order(orderParam)
    {
        // DEPRECATED.
    }

    Blob_Tensor::Blob_Tensor(mv::DataModel& dm, mv::ControlModel& cm, RelocationTable& rt , mv::Data::TensorIterator t)
    {

        int fp16_size = 2;
        this->dataType = 0;

        if (t == dm.tensorEnd())
        {
            // Exit early if this is an Empty / Null Tensor
            this->dimX = 0;
            this->dimY = 0;
            this->dimZ = 0;
            this->strideX = 0;
            this->strideY = 0;
            this->strideZ = 0;
            this->offset = 0;
            this->location = 0;
            this->dataType = 0;
            this->order = 0;
            return;
        }

        switch((int)t->getShape().ndims())
        {
            case 5:
            {
                // Hardware Weights
                this->dimX = t->getShape()[0] * t->getShape()[4];
                this->dimY = t->getShape()[1];
                this->dimZ = t->getShape()[3] * t->getShape()[2];
            }
            break;
            case 4:
            {
                // Most Software Weights
                this->dimZ = t->getShape()[3];
                this->dimY = t->getShape()[2];
                this->dimX = t->getShape()[0] * t->getShape()[1];
            }
            break;
            case 3:
            {
                // I/O
                this->dimX = t->getShape()[0];
                this->dimY = t->getShape()[1];
                this->dimZ = t->getShape()[2];
            }
            break;
            case 2:
            {
                this->dimX = 1;
                this->dimY = 1;
                this->dimZ = t->getShape()[1];
            }
            break;
            case 1:
            {
                this->dimX = t->getShape()[0];
                this->dimY = 1;
                this->dimZ = 1;
            }
            break;
            default:
            {
                std::cout << "Serialization Error: Shape of Tensor not supported in graphFile serializer" << std::endl;
                assert(0);
            }

        }


        if (!dm.hasAllocator("ConstantMemory") || !dm.hasAllocator("IntermediateMemory"))
            throw RuntimeError(dm, "Required allocators missing");

        Data::BufferIterator mem;
        mv::Control::StageIterator stg = cm.getStage(0);

        unsigned D1_stride = 0, D1_block = 0;
        unsigned D2_stride = 0, D2_block = 0;
        int block = 0;

        bool is_tight = false;

        if (t->isPopulated())
        {
            std::cout << "Populated Tensor: " << t->getName() << t->getOrder().toString()<< std::endl;
            mem = dm.getBuffer("ConstantMemory", stg, t);
            this->location = BLOB_INTERNAL_LOCATION;

            if (!mem->getStrides().empty())
            {
                std::size_t i = 1;
                for(; i < mem->getStrides().size() - 2; i++)
                {
                    D1_stride = (int)mem->getStrides()[i];
                    D1_block += (int)mem->getBlockSize();
                    if (D1_stride != 0)
                        break;
                }
                if(i >= mem->getStrides().size() - 2)
                    is_tight = true;
            }
            else
                is_tight = true;

            // CAUTION - non-tight tensors not considered here
            int rt_entry = rt.push_entry(std::pair<int, bLocation>(mem->getOffset(), bLocation::Constant ));
            this->offset = rt_entry;

        }
        else//unpopulated
        {
            bool is_external = false;

            if (t->hasAttr("modelInput"))
            {
                if (t->get<bool>("modelInput"))
                {
                    // Input tensor, non allocated in the blob
                    this->location = BLOB_INPUT_LOCATION;
                    this->offset = 0;
                    this->allocator_name = "ProgrammableInput";
                }
                else
                    throw RuntimeError(*t, "Unallocated tensor marked as non input passed for serialization");
            }
            else if (t->hasAttr("modelOutput"))
            {
                if (t->get<bool>("modelOutput"))
                {
                    // Output tensor, non allocated in the blob
                    this->location = BLOB_OUTPUT_LOCATION;
                    this->offset = 0;
                    this->allocator_name = "ProgrammableOutput";
                }
                else
                    throw RuntimeError(*t, "Unallocated tensor marked as non output passed for serialization");
            }
            else
            {
                this->location = BLOB_EXTERNAL_LOCATION;
                is_external = true;
                this->allocator_name = "IntermediateMemory";
            }

            // Will throw IndexError on incorrect stage
            mem = dm.getBuffer(allocator_name, stg, t);
            if (mem == dm.bufferEnd(allocator_name, stg))
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
                            D2_block = block;
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
                this->offset =  rt->push_entry(std::pair<int, bLocation>(mem->getOffset() + leading_pad, bLocation::Variable));

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
            switch ( t->getOrder() )
            {
                case OrderType::RowMajor: //*2 because of data_size
                {
                    if (D1_block == this->dimZ * 2)
                        local_StrideZ = D1_stride;
                    else if (D1_block == this->dimZ*this->dimY * 2)
                        local_StrideY = D1_stride;
                    else if ( D1_block == this->dimX*this->dimY*this->dimZ * 2)
                        local_StrideX = D1_stride;
                    else
                        std::cout << "Serialization Error: Cannot figure out stride translation (ColumnMajor)" << std::endl;
                }
                break;
                //PROBLEM: This assumes a 3D tensor, while weights in MX are 5D.
                //Still, weights are the same after some experiments, why is this happening?
                case OrderType::ColumnMajor:
                {
                    // This is horrible because values in allocator are 5D
                    if (D1_stride == this->dimZ* 2 * 8){
                        local_StrideY = D1_stride - 16;
                    }
                    else if (D1_stride == this->dimY*this->dimZ* 2  * 8){
                        local_StrideZ = D1_stride - 16;

                    }
                    else if ( D1_stride == this->dimX*this->dimY*this->dimZ * 2 * 8){

                        local_StrideX = D1_stride - 16;
                    }
                    else
                        std::cout << "Serialization Error: Cannot figure out stride translation (RowMajor)" << D1_block << std::endl;
                }
                break;
                case OrderType::RowMajorPlanar:
                {
                    if (D1_block == this->dimZ * 2)
                        local_StrideZ = D1_stride;
                    else if (D1_block == this->dimX*this->dimY* 2)
                        local_StrideX = D1_stride;
                    else if ( D1_block == this->dimX*this->dimY*this->dimZ * 2)
                        local_StrideY = D1_stride;
                    else
                        std::cout << "Serialization Error: Cannot figure out stride translation (RowMajorPlanar)" << std::endl;
                }
                break;
                case OrderType::ColumnMajorPlanar:
                {
                    if (D1_block == this->dimY* 2)
                        local_StrideY = D1_stride;
                    else if (D1_block == this->dimX*this->dimY* 2)
                        local_StrideX = D1_stride;
                    else if ( D1_block == this->dimX*this->dimY*this->dimZ * 2)
                        local_StrideZ = D1_stride;
                    else
                        std::cout << "Serialization Error: Cannot figure out stride translation (ColumnMajorPlanar)" << std::endl;
                }
                break;
                case OrderType::RowInterleaved: //*2 because of input_data size
                {

                    // S
                    if (D1_block == this->dimX * 2){
                        local_StrideX = D1_stride;
                    }
                    else if (D1_block == this->dimZ*this->dimX * 2){
                        local_StrideZ = D1_stride;
                    }
                    else if ( D1_block == this->dimX*this->dimY*this->dimZ * 2 )
                    {
                        local_StrideY = D1_stride;
                    }
                    else
                    {
                        std::cout << "Serialization Error: Cannot figure out stride translation (RowInterleaved) Block Size: " << D1_block << std::endl;
                    }

                    if(D2_stride != 0)
                    {
                        // Can't be X, that would be Dim1.
                        // Can't be Y, or rather, it will be unobserved in the meta info if it is.
                        // Therefore, it must be Z
                        local_StrideZ = D2_stride;
                    }
                }
                break;

                default:
                {

                }
            }

        }

        switch ( t->getOrder() )
        {
            case OrderType::RowMajorPlanar:
                {
                    if((int)t->getShape().ndims() == 3){
                        // UPA Shave
                        this->order = 0;
                        // ROW MAJOR (CHANNEL MINOR)
                        // I.E: Y, X, Z
                        this->strideZ = fp16_size;
                        this->strideX = (this->dimZ * this->strideZ) + local_StrideZ;
                        this->strideY = (this->dimX * this->strideX) + local_StrideX;
                    }else{
                        if((int)t->getShape().ndims() > 3){
                            // Software weights follow a different paradigm in c++ and python/mvtensor, causing this case.
                            // MvTensor actually uses ZYX rather than ZXY here. (confusion caused by multidimensionality)
                            this->order = 3;
                            this->strideZ = fp16_size;
                            this->strideY = (this->dimZ * this->strideZ) + local_StrideZ;
                            this->strideX = (this->dimY * this->strideY) + local_StrideY;
                        }else{
                            // Software weights follow a different paradigm in c++ and python/mvtensor, causing this case.
                            // MvTensor actually uses ZYX rather than ZXY here. (confusion caused by multidimensionality)
                            this->order = 1;
                            this->strideX = fp16_size;
                            this->strideY = (this->dimX * this->strideX) + local_StrideX;
                            this->strideZ = (this->dimY * this->strideY) + local_StrideY;
                        }
                    }
                }
                break;
            case OrderType::RowMajor:
                // Misleading - weights
                this->order = 1;
                this->strideX = fp16_size;
                this->strideY = (this->dimX * this->strideX) + local_StrideX;
                this->strideZ = (this->dimY * this->strideY) + local_StrideY;
                break;
            case OrderType::ColumnMajor:
                // NCE1 - Option 1
                // COLUMN MAJOR(NCE1 Planar)
                // I.E: X, Y, Z
                this->order = 1;    // THIS ENUM IS WRONG
                this->strideX = fp16_size;
                this->strideY = (this->dimX * this->strideX) + local_StrideX;
                this->strideZ = (this->dimY * this->strideY) + local_StrideY;
                break;
            case OrderType::ColumnMajorPlanar:
                {
                    this->order = 1;
                    this->strideY = fp16_size;
                    this->strideX = (this->dimY * this->strideY) + local_StrideY;
                    this->strideZ = (this->dimX * this->strideX) + local_StrideX;
                    // this->strideX = fp16_size;
                    // this->strideY = (this->dimX + local_StrideX)*this->strideX;
                    // this->strideZ = (this->dimY + local_StrideY)*this->strideY;
                }
                break;
             case OrderType::RowInterleaved:
                this->order = 2;
                this->strideX = fp16_size;
                this->strideZ = (this->dimX * this->strideX) + local_StrideX;
                this->strideY = (this->dimZ * this->strideZ) + local_StrideZ;
                break;

            default:
                std::cout << "Serialization Error: Order of Tensor not supported" << std::endl;
                assert(0);

        }

        // if (strcmp((*t)->getName(), "Conversion_2:0") != 0){

        // }

        // std::cout << "Order: " << (*t)->getOrder().toString() << std::endl;
        std::cout << "X: Dim:" << this->dimX << ", Stride: " << this->strideX << "(local: " << local_StrideX << ")" << std::endl;
        std::cout << "Y: Dim:" << this->dimY << ", Stride: " << this->strideY << "(local: " << local_StrideY << ")" << std::endl;
        std::cout << "Z: Dim:" << this->dimZ << ", Stride: " << this->strideZ << "(local: " << local_StrideZ << ")" << std::endl;
        // std::cout << "Block size: " << D1_block << "|" << D2_block << std::endl;
        // std::cout << "Block stride: " << D1_stride << "|" << D2_stride << std::endl;
        // std::cout << "Strides:" << D1_stride << "," << D2_stride << std::endl;
        // std::cout << "Blocks:" << D1_block << "," << D2_block << std::endl;
    }

}
