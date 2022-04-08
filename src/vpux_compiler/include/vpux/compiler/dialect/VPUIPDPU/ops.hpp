//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/ops_interfaces.hpp"

#include "vpux/compiler/dialect/VPUIPDPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/dialect.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/types.hpp"

#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/SymbolTable.h>

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUIPDPU/generated/ops.hpp.inc>

namespace vpux {
namespace VPUIPDPU {

mlir::LogicalResult verifyOp(DPUInvariant op);
mlir::LogicalResult verifyOp(DPUVariant op);
mlir::LogicalResult verifyOp(PPEFpBiasAddOp op);
mlir::LogicalResult verifyOp(PPEFpScaleMultOp op);
mlir::LogicalResult verifyOp(PPEFpConvertOp op);
mlir::LogicalResult verifyOp(PPEIntBiasAddOp op);
mlir::LogicalResult verifyOp(PPEIntScaleMultOp op);
mlir::LogicalResult verifyOp(PPEIntScaleShiftOp op);
mlir::LogicalResult verifyOp(ODUCfgOp op);
mlir::LogicalResult verifyOp(ODUSparsityOp op);
mlir::LogicalResult verifyOp(ODUOutActivationsOp op);
mlir::LogicalResult verifyOp(MPECfgOp op);

}  // namespace VPUIPDPU
}  // namespace vpux
