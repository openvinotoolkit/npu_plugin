//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

using namespace vpux;

//
// DPUTaskOp
//

void vpux::VPUIP::DPUTaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ArrayAttr start,
                                   mlir::ArrayAttr end, VPU::PaddingAttr pad, VPU::MPEMode mpeMode) {
    build(builder, state, start, end, pad, mpeMode, /*cluster_id=*/nullptr);
}
