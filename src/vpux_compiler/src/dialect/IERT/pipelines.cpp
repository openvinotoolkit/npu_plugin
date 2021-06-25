//
// Copyright Intel Corporation.
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

#include "vpux/compiler/dialect/IERT/passes.hpp"

#include <mlir/Pass/PassManager.h>

using namespace vpux;

//
// AsyncScheduling
//

void vpux::IERT::buildAsyncSchedulingPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(IERT::createWrapIntoAsyncRegionsPass(log));
    pm.addPass(IERT::createMoveViewOpsIntoAsyncRegionsPass(log));
    pm.addPass(IERT::createMoveWaitResultToAsyncBlockArgsPass(log));
    pm.addPass(IERT::createOptimizeAsyncDepsPass(log));
}

//
// registerIERTPipelines
//

void vpux::IERT::registerIERTPipelines() {
    mlir::PassPipelineRegistration<>("async-scheduling", "Asynchronous Scheduling", [](mlir::OpPassManager& pm) {
        IERT::buildAsyncSchedulingPipeline(pm);
    });
}
