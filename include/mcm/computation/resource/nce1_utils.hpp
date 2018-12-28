#ifndef NCE1_UTILS_HPP
#define NCE1_UTILS_HPP

#include "include/mcm/computation/resource/nce1.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/deployer/blob_serialization/bRelocation.hpp"

namespace mv
{
    #define BLOB_NULL_LOCATION 0
    #define BLOB_INPUT_LOCATION 1
    #define BLOB_OUTPUT_LOCATION 2
    #define BLOB_INTERNAL_LOCATION 3
    #define BLOB_EXTERNAL_LOCATION 4

    struct MXDimensionsStrides
    {
        unsigned dimX = 0;
        unsigned dimY  = 0;
        unsigned dimZ = 0;
        unsigned strideX  = 0;
        unsigned strideY = 0;
        unsigned strideZ = 0;
        unsigned dataType = 0;
        unsigned location = BLOB_NULL_LOCATION;
        bLocation blocation = bLocation::Null;
        unsigned order = 0; //MX order, kept as integer
        unsigned offset = 0;
        std::string allocator_name = "";
        bool isTight = 0;
        bool pushToRelocationTable = false;

    };

    ConvolutionParameters fillKernel2DOperationParameters(mv::Data::OpListIterator opIterator, bool add_padding = false);
    MXDimensionsStrides convertStrides(mv::Data::TensorIterator t, mv::ControlModel& cm, mv::DataModel& dm);
}

#endif
