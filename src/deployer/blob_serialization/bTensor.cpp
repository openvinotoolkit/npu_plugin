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
        // DEPRECIATED.
    }

    Blob_Tensor::Blob_Tensor(mv::DataModel* dm, mv::ControlModel* cm, RelocationTable * rt , mv::Data::TensorIterator* t)
    {

        int fp16_size = 2;
        this->dataType = 0;

        if ( t == nullptr) {  // || *t == NULL ){
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

        switch((int)(*t)->getShape().ndims())
        {
            case 5:
            {
                // Hardware Weights
                this->dimX = (*t)->getShape()[0] * (*t)->getShape()[4];
                this->dimY = (*t)->getShape()[1];
                this->dimZ = (*t)->getShape()[2] * (*t)->getShape()[3];
            }
            break;
            case 4:
            {
                // Most Software Weights
                this->dimZ = (*t)->getShape()[3];
                this->dimY = (*t)->getShape()[2];
                this->dimX = (*t)->getShape()[0] * (*t)->getShape()[1];
            }
            break;
            case 3:
            {
                // I/O
                this->dimX = (*t)->getShape()[0];
                this->dimY = (*t)->getShape()[1];
                this->dimZ = (*t)->getShape()[2];
            }
            break;
            case 2:
            {
                this->dimX = 1;
                this->dimY = 1;
                this->dimZ = (*t)->getShape()[1];
            }
            break;
            case 1:
            {
                this->dimX = (*t)->getShape()[0];
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


        try{
            if (!dm->hasAllocator("ConstantMemory") || !dm->hasAllocator("IntermediateMemory"))
                assert(0);
        }catch(mv::ArgumentError){
            printf("Serializer Warning: Allocator Missing\n");
        }

        Data::BufferIterator mem;
        mv::Control::StageIterator stg = cm->getStage(0);

        int blk_stride = 0;
        int block = 0;

        if ((*t)->isPopulated())
        {
            // std::cout << "Populated Tensor: " << (*t)->getName() << std::endl;

            mem = dm->getBuffer("ConstantMemory", stg, *t);
            this->location = BLOB_INTERNAL_LOCATION;

            if (!mem->getStrides().empty())
            {
                for(int i = 0; i != mem->getStrides().size()-1; i++)
                {
                    blk_stride = (int)mem->getStrides()[i];
                    block += (int)mem->getBlockSize();
                    if (blk_stride != 0)
                    {
                        break;
                    }
                }
            }
            else
            {
                blk_stride = -1;
            }

            int offsetValue = mem->getOffset();

            if (offsetValue % 64 != 0)
            {
                printf("Serializer Warning: Short-term alignment fix, likely cause of device crash. IMPORTANT.\n");
                offsetValue = 64+(offsetValue/64)*64 ;
            }
            int rt_entry = rt->push_entry(std::pair<int, bLocation>(offsetValue, bLocation::Constant ));
            this->offset = rt_entry;
        }
        else
        {

            mv::OpModel om(*cm);

            // std::cout << "UnPopulated Tensor: " << (*t)->getName() << std::endl;

            int no_buffers = 0;
            try{
                mem = dm->getBuffer("IntermediateMemory", stg, *t);
            }catch(mv::IndexError){
                printf("Serializer Warning: No Intermediary Buffers\n");
                no_buffers = 1;
            }

            if (no_buffers || mem == dm->bufferEnd("IntermediateMemory", stg) )
            {

                // Not Found - In or Output
                std::vector<std::string> input_names, output_names;

                for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
                {
                    if (opIterator->getOpType() == OpType::Input){
                        auto b = opIterator->getOutputTensor(0)->getName();
                        input_names.push_back(b);
                    }else if(opIterator->getOpType() == OpType::Output){
                        auto b = opIterator->getInputTensor(0)->getName();
                        output_names.push_back(b);
                    }
                }

                if(std::find(input_names.begin(), input_names.end(), (*t)->getName()) != input_names.end())
                {
                    // std::cout  << "Network Input. Note: IO Offset not supported by serializer" << std::endl;
                    this->location = BLOB_INPUT_LOCATION;
                    this->offset = 0;
                }else
                {
                    if(std::find(output_names.begin(), output_names.end(), (*t)->getName()) != output_names.end())
                    {
                        // std::cout  << "Network Output. Note: IO Offset not supported by serializer" << std::endl;
                        this->location = BLOB_OUTPUT_LOCATION;
                        this->offset = 0;
                    }
                    else
                    {
                        // std::cout << "Serialization Error: Tensor Position not resolved" << std::endl;
                        assert(0);
                    }
                }
            }
            else
            {
                // Found
                this->location = BLOB_EXTERNAL_LOCATION;
                if (!mem->getStrides().empty())
                {
                    std::cout << "Var: " << mem->toString() << std::endl;

                    // Start at 1 and go til -1 because the first and last strides are
                    // leading and trailing "padding"
                    for(int i = 1; i != mem->getStrides().size()-2; i++)
                    {
                        blk_stride = (int)mem->getStrides()[i];
                        block += (int)mem->getBlockSize();
                        if (blk_stride != 0)
                        {
                            break;
                        }
                    }
                }
                else
                {
                    blk_stride = -1;
                }
                int rt_entry = rt->push_entry(std::pair<int, bLocation>(mem->getOffset(), bLocation::Variable ));
                this->offset = rt_entry;
            }
        }


        int local_StrideX = 0;
        int local_StrideY = 0;
        int local_StrideZ = 0;

        if (blk_stride <= 0)
        {
            std::cout << "Tight" << std::endl;
            // Tight or Empty Buffer. Either way no exterior striding
        }
        else
        {
            switch ( (*t)->getOrder() )
            {
                case OrderType::ColumnMajor:
                {
                    if (block == this->dimX)
                        local_StrideX = blk_stride;
                    else if (block == this->dimX*this->dimY)
                        local_StrideY = blk_stride;
                    else if ( block == this->dimX*this->dimY*this->dimZ )
                        local_StrideZ = blk_stride;
                    else
                        std::cout << "Serialization Error: Cannot figure out stride translation (ColumnMajor)" << std::endl;
                }
                break;
                case OrderType::RowMajor:
                {
                    if (block == this->dimZ)
                        local_StrideZ = blk_stride;
                    else if (block == this->dimY*this->dimZ)
                        local_StrideY = blk_stride;
                    else if ( block == this->dimX*this->dimY*this->dimZ )
                        local_StrideX = blk_stride;
                    else
                        std::cout << "Serialization Error: Cannot figure out stride translation (RowMajor)" << std::endl;
                }
                break;
                case OrderType::RowMajorPlanar:
                {
                    if (block == this->dimZ)
                        local_StrideZ = blk_stride;
                    else if (block == this->dimX*this->dimY)
                        local_StrideX = blk_stride;
                    else if ( block == this->dimX*this->dimY*this->dimZ )
                        local_StrideY = blk_stride;
                    else
                        std::cout << "Serialization Error: Cannot figure out stride translation (RowMajorPlanar)" << std::endl;
                }
                break;
                case OrderType::ColumnMajorPlanar:
                {
                    if (block == this->dimY)
                        local_StrideY = blk_stride;
                    else if (block == this->dimX*this->dimY)
                        local_StrideX = blk_stride;
                    else if ( block == this->dimX*this->dimY*this->dimZ )
                        local_StrideZ = blk_stride;
                    else
                        std::cout << "Serialization Error: Cannot figure out stride translation (ColumnMajorPlanar)" << std::endl;
                }
                break;
                default:
                {

                }
            }
            std::cout << "local_StrideX: " << local_StrideX << std::endl;
            std::cout << "local_StrideY: " << local_StrideY << std::endl;
            std::cout << "local_StrideZ: " << local_StrideZ << std::endl;
            std::cout << "this->dimX: " << this->dimX << std::endl;
            std::cout << "this->dimY: " << this->dimY << std::endl;
            std::cout << "this->dimZ: " << this->dimZ << std::endl;
            std::cout << "Block size: " << block << std::endl;
            std::cout << "Block stride: " << blk_stride << std::endl;

        }

        switch ( (*t)->getOrder() )
        {
            case OrderType::RowMajor:
                // UPA Shave
                this->order = 0;
                // ROW MAJOR (CHANNEL MINOR)
                // I.E: Y, X, Z
                this->strideZ = fp16_size;
                this->strideX = (this->dimZ + local_StrideZ)*this->strideZ;
                this->strideY = (this->dimX + local_StrideX)*this->strideX;
                break;
            case OrderType::RowMajorPlanar:
                // NCE1 - Option 1
                // ROW MAJOR PLANAR (PLANAR)
                // I.E: Z, Y, X
                this->order = 1;
                this->strideX = fp16_size;
                this->strideY = (this->dimX + local_StrideX)*this->strideX;
                this->strideZ = (this->dimY + local_StrideY)*this->strideY;
                break;
            case OrderType::ColumnMajor:
                // NCE1 - Option 2
                // COLUMN MAJOR(INTERLEAVED)
                // I.E: X, Z, Y
                this->order = 2;
                this->strideX = fp16_size;
                this->strideZ = (this->dimX + local_StrideX)*this->strideX;
                this->strideY = (this->dimZ + local_StrideZ)*this->strideZ;
                break;
            case OrderType::ColumnMajorPlanar:
                this->order = 3;
                this->strideZ = fp16_size;
                this->strideY = (this->dimZ + local_StrideZ)*this->strideZ;
                this->strideX = (this->dimY + local_StrideY)*this->strideY;
                break;
            default:
                std::cout << "Serialization Error: Order of Tensor not supported" << std::endl;
                assert(0);
        }
    }

}