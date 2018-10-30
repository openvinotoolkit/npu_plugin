#ifndef NCE1_UTILS_HPP
#define NCE1_UTILS_HPP

#include "mcm/computation/resource/nce1.hpp"
#include "mcm/computation/op/op.hpp"

namespace mv
{
    mv::ConvolutionParameters fillConvolutionParameters(mv::Data::OpListIterator convIterator, bool add_padding = false);
}

#endif
