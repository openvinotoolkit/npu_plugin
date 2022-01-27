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

#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"

using namespace vpux;

void vpux::VPUIPRegMapped::PPETaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state,
                                            vpux::VPUIPRegMapped::PPELayerType ppe_layer_type, int32_t clamp_low,
                                            int32_t clamp_high, int32_t lrelu_mult, uint32_t lrelu_shift) {
    build(builder, state, VPUIPRegMapped::PPELayerTypeAttr::get(builder.getContext(), ppe_layer_type),
          builder.getI32IntegerAttr(clamp_low), builder.getI32IntegerAttr(clamp_high),
          builder.getI32IntegerAttr(lrelu_mult), builder.getUI32IntegerAttr(lrelu_shift));
}
