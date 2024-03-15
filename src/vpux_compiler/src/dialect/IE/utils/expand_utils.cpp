//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/expand_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/utils/core/logger.hpp"

namespace vpux {
namespace IE {

// With respect to eltwise ops in a chain, for example:
//   Expand -> Add -> Slice -> Expand -> Add -> Slice
// It will be beneficial to keep the 2nd Expand for the 2nd Add instead of folding with Slice.
// So that the 2nd Add can utilize AdjustInputShapeForEltwise pass
bool beneficialToKeepExpand(ShapeRef unExpandedShape, ShapeRef expandedShape, mlir::Operation* childOp) {
    if (!childOp->hasOneUse()) {
        return false;
    }
    const auto isEltwiseOp = [](mlir::Operation* op) {
        if (op == nullptr) {
            return false;
        }
        // Mul/Sub/Add are selected since they are covered by the AdjustInputShapeForEltwise pass
        if (auto grpConvOp = mlir::dyn_cast<IE::GroupConvolutionOp>(op)) {
            return groupConvIsEltwise(grpConvOp);
        } else if (mlir::isa<IE::MultiplyOp, IE::SubtractOp, IE::AddOp>(op)) {
            return true;
        }
        return false;
    };

    vpux::Logger log("beneficialToKeepExpand", vpux::LogLevel::Info);
    while (isEltwiseOp(childOp) && VPU::NCEInvariant::isSupported(childOp).succeeded()) {
        auto shapeCastResult = getShapeCastExpandedShape(childOp, expandedShape, unExpandedShape, log);
        if (mlir::failed(shapeCastResult)) {
            return false;
        }
        auto sliceChildOp = mlir::dyn_cast_or_null<IE::SliceOp>(*childOp->getResult(0).getUsers().begin());
        if (sliceChildOp == nullptr) {
            return true;
        }
        auto expandChildOp = mlir::dyn_cast_or_null<IE::ExpandOp>(*sliceChildOp->getResult(0).getUsers().begin());
        if (expandChildOp == nullptr) {
            return true;
        }
        childOp = *childOp->getResult(0).getUsers().begin();
        if (childOp == nullptr) {
            return true;
        } else if (!childOp->hasOneUse()) {
            return false;
        }
    }
    return false;
}

}  // namespace IE
}  // namespace vpux
