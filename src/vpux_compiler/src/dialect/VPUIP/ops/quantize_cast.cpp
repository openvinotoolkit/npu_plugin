//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

using namespace vpux;

mlir::Value VPUIP::QuantizeCastOp::getViewSource() {
    return input();
}

mlir::OpFoldResult VPUIP::QuantizeCastOp::fold(ArrayRef<mlir::Attribute>) {
    return input().getType() == output().getType() ? input() : nullptr;
}
