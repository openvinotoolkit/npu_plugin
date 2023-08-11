//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IERT/ops.hpp"

mlir::Value vpux::IERT::QuantizeCastOp::getViewSource() {
    return input();
}

mlir::OpFoldResult vpux::IERT::QuantizeCastOp::fold(ArrayRef<mlir::Attribute>) {
    return input().getType() == output().getType() ? input() : nullptr;
}
