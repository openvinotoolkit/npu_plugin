//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

void vpux::VPUIP::CopyOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                mlir::Value output_buff) {
    build(builder, state, input, output_buff, /*channelType=*/nullptr, /*spillId=*/nullptr);
}

void vpux::VPUIP::CopyOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Type output,
                                mlir::Value input, mlir::Value output_buff) {
    build(builder, state, output, input, output_buff, /*channelType=*/nullptr, /*spillId=*/nullptr);
}

void vpux::VPUIP::CopyOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                mlir::Value output_buff, int64_t spillId) {
    build(builder, state, input, output_buff, /*channelType=*/nullptr, vpux::getIntAttr(builder, spillId));
}
