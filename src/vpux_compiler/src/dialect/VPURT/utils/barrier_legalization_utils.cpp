//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"

using namespace vpux;

void VPURT::postProcessBarrierOps(mlir::func::FuncOp func) {
    // move barriers to top and erase unused
    auto barrierOps = to_small_vector(func.getOps<VPURT::DeclareVirtualBarrierOp>());
    auto& block = func.getBody().front();

    VPURT::DeclareVirtualBarrierOp prevBarrier = nullptr;
    for (auto& barrierOp : barrierOps) {
        // remove barriers with no use
        if (barrierOp.getBarrier().use_empty()) {
            barrierOp->erase();
            continue;
        }

        // move barriers to top of block
        if (prevBarrier != nullptr) {
            barrierOp->moveAfter(prevBarrier);
        } else {
            barrierOp->moveBefore(&block, block.begin());
        }

        prevBarrier = barrierOp;
    }
}
