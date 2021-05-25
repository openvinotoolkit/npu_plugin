//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
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
