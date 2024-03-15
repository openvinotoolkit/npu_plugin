//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops_interfaces.hpp"

using namespace vpux;

VPURegMapped::TaskType VPURegMapped::FetchTaskOp::getTaskType() {
    return VPURegMapped::TaskType::DMA;
}

void VPURegMapped::FetchTaskOp::setTaskLocation(mlir::Value) {
    return;
}

mlir::Value VPURegMapped::FetchTaskOp::getTaskLocation() {
    return nullptr;
}
