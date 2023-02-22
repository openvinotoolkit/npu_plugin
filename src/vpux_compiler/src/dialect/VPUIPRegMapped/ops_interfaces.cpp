//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIPRegMapped/ops_interfaces.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/format.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

//
// SingleOutputAsIndexOp
//

mlir::LogicalResult vpux::VPUIPRegMapped::verifySingleOutputAsIndexOp(mlir::Operation* op) {
    if (op->getNumResults() != 1) {
        return errorAt(op, "Operation '{0}' does not have a single index type result", op->getName());
    }
    if (!op->getResult(0).getType().isa<VPUIPRegMapped::IndexType>()) {
        return errorAt(op, "Operation '{0}' result type is not VPUIPRegMapped::IndexType", op->getName());
    }

    return mlir::success();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUIPRegMapped/generated/ops_interfaces.cpp.inc>
