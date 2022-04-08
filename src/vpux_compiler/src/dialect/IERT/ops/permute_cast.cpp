//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IERT/ops.hpp"

using namespace vpux;

mlir::Value vpux::IERT::PermuteCastOp::getViewSource() {
    return source();
}
