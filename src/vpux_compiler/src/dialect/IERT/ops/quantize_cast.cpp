//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IERT/ops.hpp"

mlir::Value vpux::IERT::QuantizeCastOp::getViewSource() {
    return getInput();
}

mlir::OpFoldResult vpux::IERT::QuantizeCastOp::fold(FoldAdaptor) {
    return getInput().getType() == getOutput().getType() ? getInput() : mlir::TypedValue<mlir::MemRefType>{nullptr};
}
