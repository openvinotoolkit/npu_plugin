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

mlir::FailureOr<int64_t> getConstOrAttrValue(mlir::Value input, mlir::IntegerAttr attr) {
    if (input) {
        auto op = input.getDefiningOp<Const::DeclareOp>();
        if (op == nullptr) {
            return mlir::failure();
        }

        const auto content = op.getContent();
        if (!content.isSplat()) {
            return mlir::failure();
        }

        return content.getSplatValue<int64_t>();
    }

    return attr.getValue().getSExtValue();
}

}  // namespace vpux
