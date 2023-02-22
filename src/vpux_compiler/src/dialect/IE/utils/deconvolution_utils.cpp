//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/deconvolution_utils.hpp"

namespace vpux {
namespace IE {
// Checks whether the Deconvolution filter is a constant or a FakeQuantize with a constant input
mlir::FailureOr<Const::DeclareOp> getConstFilter(IE::DeconvolutionOp deconv) {
    if (auto filterFq = deconv.filter().getDefiningOp<IE::FakeQuantizeOp>()) {
        if (auto filterConst = filterFq.input().getDefiningOp<Const::DeclareOp>()) {
            return filterConst;
        }
    } else if (auto filterConst = deconv.filter().getDefiningOp<Const::DeclareOp>()) {
        return filterConst;
    }
    return mlir::failure();
}

mlir::LogicalResult canConvertDeconvToConv(IE::DeconvolutionOp deconv) {
    if (getShape(deconv.feature()).size() != 4) {
        return mlir::failure();
    }

    if (mlir::failed(IE::getConstFilter(deconv))) {
        return mlir::failure();
    }

    return mlir::success();
}
}  // namespace IE
}  // namespace vpux
