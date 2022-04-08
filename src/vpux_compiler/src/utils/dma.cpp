//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/utils/dma.hpp"

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

int64_t vpux::getDMAPortValue(mlir::Operation* wrappedTaskOp) {
    if (auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(wrappedTaskOp)) {
        return dmaOp.port();
    } else if (auto compressedDmaOp = mlir::dyn_cast<VPUIP::CompressedDMAOp>(wrappedTaskOp)) {
        return compressedDmaOp.port();
    } else if (auto depthToSpaceDMAOp = mlir::dyn_cast<VPUIP::DepthToSpaceDMAOp>(wrappedTaskOp)) {
        return depthToSpaceDMAOp.port();
    } else if (auto spaceToDepthDMAOp = mlir::dyn_cast<VPUIP::SpaceToDepthDMAOp>(wrappedTaskOp)) {
        return spaceToDepthDMAOp.port();
    } else if (auto perAxisTileDMAOp = mlir::dyn_cast<VPUIP::PerAxisTileDMAOp>(wrappedTaskOp)) {
        return perAxisTileDMAOp.port();
    } else if (auto permuteDMAOp = mlir::dyn_cast<VPUIP::PermuteDMAOp>(wrappedTaskOp)) {
        return permuteDMAOp.port();
    } else if (auto expandDMAOp = mlir::dyn_cast<VPUIP::ExpandDMAOp>(wrappedTaskOp)) {
        return expandDMAOp.port();
    } else if (auto upsamplingDMAOp = mlir::dyn_cast<VPUIP::UpsamplingDMAOp>(wrappedTaskOp)) {
        return upsamplingDMAOp.port();
    }

    VPUX_THROW("Could not cast to DMA task");
}
