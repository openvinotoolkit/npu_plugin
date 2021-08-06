#ifndef PASS_DTYPE_UTILS_HPP_
#define PASS_DTYPE_UTILS_HPP_

#include <set>
#include <include/mcm/tensor/dtype/dtype.hpp>

namespace mv
{

static const std::set<std::pair<mv::DType, mv::DType>> supportedConversions = {
    std::make_pair(mv::DType("Float16"), mv::DType("Float32")),
    std::make_pair(mv::DType("Float32"), mv::DType("Float16")),
    std::make_pair(mv::DType("Float16"), mv::DType("Int32")),
    std::make_pair(mv::DType("Int32"),   mv::DType("Float16")),
    std::make_pair(mv::DType("Float16"), mv::DType("UInt8")),
    std::make_pair(mv::DType("UInt8"),   mv::DType("Float16")),
    std::make_pair(mv::DType("UInt8"),   mv::DType("Float32")),
    std::make_pair(mv::DType("Float32"), mv::DType("UInt8")),
    std::make_pair(mv::DType("Int32"),   mv::DType("UInt8")),
//  std::make_pair(mv::DType("UInt8"),   mv::DType("Int32")),   // not supported by SW kernel
};

}

#endif // PASS_DTYPE_UTILS_HPP_
