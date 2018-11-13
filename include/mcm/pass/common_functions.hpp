#ifndef MV_ADAPTATION_COMMON_FUNCTIONS_
#define MV_ADAPTATION_COMMON_FUNCTIONS_

#include "mcm/computation/op/op.hpp"
#include "meta/include/mcm/op_model.hpp"

namespace mv
{
    Data::OpListIterator linkNewOperations(Data::OpListIterator parentOpIt, Data::TensorIterator sourceTensor, OpModel om, Data::OpListIterator opIt);
}

#endif
