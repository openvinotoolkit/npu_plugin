//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// AsyncScheduling
//

void vpux::VPUIP::buildAsyncSchedulingPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(createMoveDeclarationsToTopPass(log));
    pm.addPass(VPUIP::createWrapIntoAsyncRegionsPass(log));
    pm.addPass(VPUIP::createMoveViewOpsIntoAsyncRegionsPass(log));
    pm.addPass(VPUIP::createMoveWaitResultToAsyncBlockArgsPass(log));
}

//
// HardwareAdaptation
//

void vpux::VPUIP::buildHardwareAdaptationPipeline(mlir::OpPassManager& pm,
                                                  const VPUIP::HardwareAdaptationOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(VPUIP::createConvertTransferOpsToDMAsPass(log));
    pm.addPass(VPUIP::createConvertAllocationsToDeclarationsPass(log));
    pm.addPass(VPUIP::createConvertViewOpsToDeclarationsPass(log));
    if (options.enableCompressWeights) {
        pm.addPass(VPUIP::createCompressWeightsPass(log));
    }
    pm.addPass(VPUIP::createConvertAsyncOpsToTasksPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(createMoveDeclarationsToTopPass(log));
}

//
// registerVPUIPPipelines
//

void VPUIP::registerVPUIPPipelines() {
    mlir::PassPipelineRegistration<>("async-scheduling", "Asynchronous Scheduling", [](mlir::OpPassManager& pm) {
        VPUIP::buildAsyncSchedulingPipeline(pm);
    });

    mlir::PassPipelineRegistration<VPUIP::HardwareAdaptationOptions>(
            "hardware-adaptation", "Hardware Adaptation",
            [](mlir::OpPassManager& pm, const VPUIP::HardwareAdaptationOptions& options) {
                VPUIP::buildHardwareAdaptationPipeline(pm, options);
            });
}
