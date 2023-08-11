//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

using namespace vpux;

//
// DPUTaskOp
//

void vpux::VPUIP::DPUTaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ArrayAttr outStart,
                                   mlir::ArrayAttr outEnd, VPU::PaddingAttr pad, VPU::MPEMode mpeMode) {
    build(builder, state, outStart, outEnd, /*inStart=*/nullptr, /*inEnd=*/nullptr, pad, mpeMode,
          /*cluster_id=*/nullptr, /* workload_id =  */ nullptr);
}

void vpux::VPUIP::DPUTaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ArrayAttr outStart,
                                   mlir::ArrayAttr outEnd, VPU::PaddingAttr pad, VPU::MPEMode mpeMode,
                                   mlir::IntegerAttr clusterId) {
    build(builder, state, outStart, outEnd, /*inStart=*/nullptr, /*inEnd=*/nullptr, pad, mpeMode, clusterId,
          /* workload_id =  */ nullptr);
}

void vpux::VPUIP::DPUTaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ArrayAttr outStart,
                                   mlir::ArrayAttr outEnd, mlir::ArrayAttr inStart, mlir::ArrayAttr inEnd,
                                   VPU::PaddingAttr pad, VPU::MPEMode mpeMode) {
    build(builder, state, outStart, outEnd, inStart, inEnd, pad, mpeMode,
          /*cluster_id=*/nullptr, /* workload_id =  */ nullptr);
}

void vpux::VPUIP::DPUTaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ArrayAttr outStart,
                                   mlir::ArrayAttr outEnd, mlir::ArrayAttr inStart, mlir::ArrayAttr inEnd,
                                   VPU::PaddingAttr pad, VPU::MPEMode mpeMode, mlir::IntegerAttr clusterId) {
    build(builder, state, outStart, outEnd, inStart, inEnd, pad, mpeMode, clusterId, /* workload_id =  */ nullptr);
}
