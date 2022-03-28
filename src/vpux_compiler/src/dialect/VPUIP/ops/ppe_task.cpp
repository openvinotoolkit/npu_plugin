//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

using namespace vpux;

void vpux::VPUIP::PPETaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, VPU::PPEMode ppe_layer_type,
                                   int64_t clamp_low, int64_t clamp_high, int64_t lrelu_mult, int64_t lrelu_shift) {
    build(builder, state, VPU::PPEModeAttr::get(builder.getContext(), ppe_layer_type),
          builder.getI64IntegerAttr(clamp_low), builder.getI64IntegerAttr(clamp_high),
          builder.getI64IntegerAttr(lrelu_mult), builder.getI64IntegerAttr(lrelu_shift), nullptr, nullptr, nullptr);
}

void vpux::VPUIP::PPETaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, VPU::PPEMode ppe_layer_type,
                                   int64_t clamp_low, int64_t clamp_high, int64_t lrelu_mult, int64_t lrelu_shift,
                                   int64_t quant_mult, int64_t quant_shift) {
    build(builder, state, VPU::PPEModeAttr::get(builder.getContext(), ppe_layer_type),
          builder.getI64IntegerAttr(clamp_low), builder.getI64IntegerAttr(clamp_high),
          builder.getI64IntegerAttr(lrelu_mult), builder.getI64IntegerAttr(lrelu_shift),
          builder.getI64ArrayAttr({quant_mult}), builder.getI64ArrayAttr({quant_shift}), builder.getI64IntegerAttr(0));
}
