//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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
