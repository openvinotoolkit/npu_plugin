//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/attributes/enums.hpp"
#include "vpux/compiler/dialect/IE/attributes/structs.hpp"
#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

//
// Generated
//

#include <vpux/compiler/dialect/IE/generated/dialect.hpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/IE/generated/ops.hpp.inc>

//
// Operation verifiers
//

namespace vpux {
namespace IE {

bool isActShaveKernel(mlir::Operation* operation);

mlir::LogicalResult verifyOp(CNNNetworkOp op);
mlir::LogicalResult verifyOp(BucketizeOp op);
mlir::LogicalResult verifyOp(DataInfoOp op);
mlir::LogicalResult verifyOp(ExpandOp op);
mlir::LogicalResult verifyOp(GatherNDOp op);
mlir::LogicalResult verifyOp(GridSampleOp op);
mlir::LogicalResult verifyOp(ReduceSumOp op);
mlir::LogicalResult verifyOp(AffineReshapeOp op);
mlir::LogicalResult verifyOp(ClampOp op);

//
// Tiling
//

// Adjust paddings attributes for tiled input
template <typename ConcreteOp>
void adjustPaddings(ConcreteOp* op, const TilingInfo& inputTiling) {
    const auto& inputTilePads = inputTiling.pads;
    VPUX_THROW_UNLESS(inputTilePads.hasValue(), "Missing tile information for paddings");

    const std::array<int64_t, 2> padsBegin = {inputTilePads->top, inputTilePads->left};
    const std::array<int64_t, 2> padsEnd = {inputTilePads->bottom, inputTilePads->right};

    auto newPadsBeginAttr = getIntArrayAttr(op->getContext(), padsBegin);
    auto newPadsEndAttr = getIntArrayAttr(op->getContext(), padsEnd);

    op->pads_beginAttr(newPadsBeginAttr);
    op->pads_endAttr(newPadsEndAttr);
}

}  // namespace IE
}  // namespace vpux
