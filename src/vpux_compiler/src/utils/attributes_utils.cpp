//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/attributes_utils.hpp"

namespace vpux {

int64_t getPositiveAxisInd(mlir::IntegerAttr axisIndAttr, int64_t rank) {
    auto axis = axisIndAttr.getValue().getSExtValue();

    if (axis < 0) {
        axis += rank;
    }

    return axis;
}

}  // namespace vpux
