//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

using namespace vpux;

//
// SparsityInfoOp
//

double vpux::IE::SparsityInfoOp::getRatio() {
    return ratioAttr().getValueAsDouble();
}
