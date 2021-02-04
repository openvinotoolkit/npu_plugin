//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "vpux/compiler/core/attributes/dim.hpp"

using namespace vpux;

//
// DimBase
//

void vpux::details::validateDimAttrs(StringRef className, int32_t ind) {
    VPUX_THROW_UNLESS(ind >= 0, "Got negative index {0} for {1}", ind, className);

    VPUX_THROW_UNLESS(static_cast<size_t>(ind) < MAX_NUM_DIMS, "{0} index {1} exceeds maximal supported value {2}",
                      className, ind, MAX_NUM_DIMS);
}

//
// Dim
//

StringRef vpux::Dim::getClassName() {
    return "Dim";
}

//
// MemDim
//

StringRef vpux::MemDim::getClassName() {
    return "MemDim";
}
