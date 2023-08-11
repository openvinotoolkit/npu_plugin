//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/EMU/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// AdjustForEMU
//

void vpux::EMU::buildAdjustForEMU(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();
    pm.addPass(EMU::createSqueezeBiasShapePass(log));
    pm.addPass(EMU::createAdjustFQPrecisionPass(log));
    pm.addPass(createConvertVPUNCEToEMUPass(log));
    pm.addPass(EMU::createRemoveWeightsAlignmentPass(log));
    pm.addPass(EMU::createAddWeightsTableToEmuPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// registerEMUPipelines
//

void vpux::EMU::registerEMUPipelines() {
    mlir::PassPipelineRegistration<>("adjust-EMU", "Adjust IR for EMU target", [](mlir::OpPassManager& pm) {
        EMU::buildAdjustForEMU(pm);
    });
}
