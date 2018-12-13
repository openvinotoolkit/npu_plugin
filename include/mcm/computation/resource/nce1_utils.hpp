#ifndef NCE1_UTILS_HPP
#define NCE1_UTILS_HPP

#include "include/mcm/computation/resource/nce1.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"

namespace mv
{
    mv::ConvolutionParameters fillKernel2DOperationParameters(mv::Data::OpListIterator opIterator, bool add_padding = false);
}

#endif
