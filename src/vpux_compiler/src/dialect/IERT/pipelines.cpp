//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IERT/passes.hpp"

#include "vpux/compiler/core/passes.hpp"

#include <mlir/Pass/PassManager.h>

using namespace vpux;

//
// AsyncScheduling
//

void vpux::IERT::buildAsyncSchedulingPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(createMoveDeclarationsToTopPass(log));
    pm.addPass(IERT::createWrapIntoAsyncRegionsPass(log));
    pm.addPass(IERT::createMoveViewOpsIntoAsyncRegionsPass(log));
    pm.addPass(IERT::createMoveWaitResultToAsyncBlockArgsPass(log));
}

//
// registerIERTPipelines
//

void vpux::IERT::registerIERTPipelines() {
    mlir::PassPipelineRegistration<>("async-scheduling", "Asynchronous Scheduling", [](mlir::OpPassManager& pm) {
        IERT::buildAsyncSchedulingPipeline(pm);
    });
}
