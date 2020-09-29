#include "include/mcm/tensor/tensor_info.hpp"

namespace mv {

std::string TensorInfo::toString() const
{
    return "[" + shape_.toString() + ", " + type_.toString() + ", " +
         order_.toString() + "]";
}

std::string TensorInfo::getLogID() const
{
    return "TensorInfo:" + toString();
}

}
