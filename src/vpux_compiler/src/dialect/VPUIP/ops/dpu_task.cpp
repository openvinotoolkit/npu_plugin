//
// Copyright 2022 Intel Corporation.
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

//
// NNDMAOp
//

void vpux::VPUIP::DPUTaskOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ArrayAttr start,
                                   mlir::ArrayAttr end, VPU::PaddingAttr pad, VPU::MPEMode mpeMode) {
    build(builder, state, start, end, pad, mpeMode, /*cluster_id=*/nullptr);
}
