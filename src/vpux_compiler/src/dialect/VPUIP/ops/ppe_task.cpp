//
// Copyright 2021 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
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
