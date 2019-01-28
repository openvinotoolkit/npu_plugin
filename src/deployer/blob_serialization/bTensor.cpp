#include <stdio.h>
#include "include/mcm/deployer/blob_serialization/bTensor.hpp"
#include "include/mcm/target/myriadx/nce1_utils.hpp"
#include <string.h>

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

        MXDimensionsStrides mxDimensionsStrides = convertStrides(t, cm, dm);

        this->dimX = mxDimensionsStrides.dimX;
        this->dimY = mxDimensionsStrides.dimY;
        this->dimZ = mxDimensionsStrides.dimZ;
        this->strideX = mxDimensionsStrides.strideX;
        this->strideY = mxDimensionsStrides.strideY;
        this->strideZ = mxDimensionsStrides.strideZ;
        this->location = mxDimensionsStrides.location;
        this->dataType = mxDimensionsStrides.dataType;
        this->order = mxDimensionsStrides.order;

        if(mxDimensionsStrides.pushToRelocationTable)
            this->offset = rt.push_entry(std::pair<int, bLocation>(mxDimensionsStrides.offset, mxDimensionsStrides.blocation));
        else
            this->offset = 0;

    }

}
