#ifndef NCE1_UTILS_HPP
#define NCE1_UTILS_HPP

#include "mcm/computation/resource/nce1.hpp"
#include "mcm/computation/op/computation_op.hpp"

namespace mv
{
    mv::ConvolutionParameters fillKernel2DOperationParameters(mv::Data::OpListIterator opIterator, bool add_padding = false);
}

#endif
