//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/StandardOps/Transforms/Passes.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// LowerIE2IERT
//

void vpux::buildLowerIE2IERTPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(createBufferizeIEPass(log));
    pm.addPass(createBufferizeFuncAndReturnPass(log));
    pm.addPass(createAddBuffersForNetResults(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// LowerIERT2VPUIP
//

void vpux::buildLowerIERT2VPUIPPipeline(mlir::OpPassManager& pm, const LowerIERT2VPUIPOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(createConvertLayers2VPUIPPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(createConvertDeclarations2VPUIPPass(log));
    pm.addPass(createConvertViewOps2VPUIPPass(log));
    if (options.enableCompressWeights) {
        pm.addPass(vpux::VPUIP::createCompressWeightsPass(log));
    }
    pm.addPass(createConvertAsyncOps2VPUIPPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(createMoveDeclarationsToTopPass(log));
}

//
// registerConversionPipelines
//

void vpux::registerConversionPipelines() {
    mlir::PassPipelineRegistration<>("lower-IE-to-IERT", "Performs full lowering from the IE Dialect to IERT Dialect",
                                     [](mlir::OpPassManager& pm) {
                                         buildLowerIE2IERTPipeline(pm);
                                     });

    mlir::PassPipelineRegistration<LowerIERT2VPUIPOptions>(
            "lower-IERT-to-VPUIP", "Performs full lowering from the IERT Dialect to VPUIP Dialect",
            [](mlir::OpPassManager& pm, const LowerIERT2VPUIPOptions& options) {
                buildLowerIERT2VPUIPPipeline(pm, options);
            });
}
