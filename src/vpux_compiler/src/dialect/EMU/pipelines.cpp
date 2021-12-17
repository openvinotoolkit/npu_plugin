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
    pm.addPass(mlir::createCanonicalizerPass(grc));
}


//
// registerEMUPipelines
//

void vpux::EMU::registerEMUPipelines() {
    mlir::PassPipelineRegistration<>(
            "adjust-EMU", "Adjust IR for EMU target",
            [](mlir::OpPassManager& pm) {
                EMU::buildAdjustForEMU(pm);
            });
}
