//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIPRegMapped/utils.hpp"

namespace vpux {
namespace VPUIPRegMapped {

std::pair<uint8_t, uint32_t> getMaxVID(mlir::Operation::operand_range range) {
    uint32_t maxBarrierVID = 0;
    uint8_t maxBarrierVidPid = 0;

    for (auto bar : range) {
        if (auto waitBar = mlir::dyn_cast<VPUIPRegMapped::ConfigureBarrierOp>(bar.getDefiningOp())) {
            auto waitBarVID = waitBar.getType().getValue();
            if (waitBarVID > maxBarrierVID) {
                maxBarrierVID = waitBarVID;
                maxBarrierVidPid = waitBar.id();
            }
        }
    }

    return {maxBarrierVidPid, maxBarrierVID};
}

uint64_t computeMask(mlir::Operation::operand_range barriers) {
    uint64_t mask = 0;
    for (auto barrier : barriers) {
        mask |= static_cast<uint64_t>(1)
                << mlir::cast<VPUIPRegMapped::ConfigureBarrierOp>(barrier.getDefiningOp()).id();
    }
    return mask;
}

}  // namespace VPUIPRegMapped
}  // namespace vpux
