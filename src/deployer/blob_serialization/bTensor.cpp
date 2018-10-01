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

        if ( t == nullptr)
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


        if (!dm->hasAllocator("ConstantMemory") || !dm->hasAllocator("IntermediateMemory"))
            throw RuntimeError(*dm, "Required allocators missing");

        Data::BufferIterator mem;
        mv::Control::StageIterator stg = cm->getStage(0);

        int blk_stride = 0;
        int block = 0;

        if ((*t)->isPopulated())
        {

            mem = dm->getBuffer("ConstantMemory", stg, *t);
            this->location = BLOB_INTERNAL_LOCATION;

            if (!mem->getStrides().empty())
            {
                for(std::size_t i = 1; i < mem->getStrides().size() - 2; i++)
                {
                    blk_stride = (int)mem->getStrides()[i];
                    block += (int)mem->getBlockSize();
                    if (blk_stride != 0)
                    {
                        break;
                    }
                }
                this->dimY = this->dimY + 1;

            }
            else
            {
                blk_stride = -1;
            }

            // CAUTION - non-tight tensors not considered here
            int rt_entry = rt->push_entry(std::pair<int, bLocation>(mem->getOffset(), bLocation::Constant ));
            this->offset = rt_entry;

        }
        else
        {

            mv::OpModel om(*cm);

            if ((*t)->hasAttr("modelInput"))
            {
                if ((*t)->get<bool>("modelInput"))
                {
                    // Input tensor, non allocated in the blob
                    this->location = BLOB_INPUT_LOCATION;
                    this->offset = 0;
                }
                else
                    throw RuntimeError(**t, "Unallocated tensor marked as non input passed for serialization");
            }
            else if ((*t)->hasAttr("modelOutput"))
            {
                if ((*t)->get<bool>("modelOutput"))
                {
                    // Output tensor, non allocated in the blob
                    this->location = BLOB_OUTPUT_LOCATION;
                    this->offset = 0;
                }
                else
                    throw RuntimeError(**t, "Unallocated tensor marked as non output passed for serialization");
            }
            else
            {
                // Will throw IndexError on incorrect stage
                mem = dm->getBuffer("IntermediateMemory", stg, *t);
                if (mem == dm->bufferEnd("IntermediateMemory", stg))
                    throw RuntimeError(**t, "Unallocated tensor found during the serialization");

                this->location = BLOB_EXTERNAL_LOCATION;
                unsigned leading_pad = 0;
                if (!mem->getStrides().empty())
                {

                    // Start at 1 and go til -1 because the first and last strides are
                    // leading and trailing "padding"
                    for(std::size_t i = 1; i < mem->getStrides().size() - 2; i++)
                    {

                        blk_stride = (int)mem->getStrides()[i];
                        block += (int)mem->getBlockSize();
                        if (blk_stride != 0)
                            break;

                    }

                    leading_pad = mem->getStrides()[0];

                }
                else
                    blk_stride = -1;

                this->offset =  rt->push_entry(std::pair<int, bLocation>(mem->getOffset() + leading_pad, bLocation::Variable));

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
            case OrderType::RowMajorPlanar:
                // UPA Shave
                this->order = 0;
                // ROW MAJOR (CHANNEL MINOR)
                // I.E: Y, X, Z
                this->strideZ = fp16_size;
                this->strideX = (this->dimZ + local_StrideZ)*this->strideZ;
                this->strideY = (this->dimX + local_StrideX)*this->strideX;
                break;
            case OrderType::RowMajor:
                this->order = 2;
                this->strideX = fp16_size;
                this->strideY = (this->dimX + local_StrideX)*this->strideX;
                this->strideZ = (this->dimY + local_StrideY)*this->strideY;
                break;
            case OrderType::ColumnMajor:
                // NCE1 - Option 1
                // COLUMN MAJOR(NCE1 Planar)
                // I.E: X, Y, Z
                this->order = 1;
                this->strideX = fp16_size;
                this->strideY = (this->dimX + local_StrideX)*this->strideX;
                this->strideZ = (this->dimY + local_StrideY)*this->strideY;
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