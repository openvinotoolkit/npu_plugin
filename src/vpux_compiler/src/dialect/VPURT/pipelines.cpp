//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU37XX/dialect/VPURT/passes.hpp"
#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/dialect/VPURT/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// BarrierLegalization
//

void vpux::VPURT::buildBarrierLegalizationPipeline(mlir::OpPassManager& pm,
                                                   const VPURT::BarrierLegalizationOptions& options, Logger log) {
    if (options.simpleSchedule) {
        pm.addPass(VPURT::createSimplifySchedulePass(options.shareWaitAndUpdateBarriers,
                                                     options.reduceParallelControlFlows, log));
    }
    pm.addPass(VPURT::createSplitExceedingVariantCountBarriersPass(log));
    pm.addPass(VPURT::createSatisfyOneWaitBarrierPerTaskPass(log));
    pm.addPass(VPURT::createReduceExceedingActiveCountBarriersPass(log));
    pm.addPass(VPURT::arch37xx::createAddUpdateBarrierForSwKernelsPass(log));
}

//
// registerVPURTPipelines
//

void VPURT::registerVPURTPipelines() {
    mlir::PassPipelineRegistration<VPURT::BarrierLegalizationOptions>(
            "barrier-legalization", "Barrier Legalization",
            [](mlir::OpPassManager& pm, const VPURT::BarrierLegalizationOptions& options) {
                VPURT::buildBarrierLegalizationPipeline(pm, options);
            });
}
