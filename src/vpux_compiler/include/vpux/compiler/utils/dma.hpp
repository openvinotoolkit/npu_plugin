//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

namespace vpux {

int64_t getDMAPortValue(mlir::Operation* wrappedTaskOp);

// In VPU different DMA channel receives data movement job from link agent.
// Different channel is used based on transaction source
// - DDR channel - DMA SRC is DDR
// - CMX channel - DMA SRC is CMX or HW register
VPUIP::DmaChannelType setDMAChannelType(VPUIP::DMATypeOpInterface dmaOp, VPU::ArchKind arch);

/// @brief Checks if the ConvertOp is supported on DMA
/// @param convertOp template argument
/// @return boolean

template <typename T>
bool isConvertSupportedOnDMA(T /* convertOp */) {
    return false;
}
}  // namespace vpux
