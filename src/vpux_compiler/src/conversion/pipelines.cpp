//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/dialect/ELF/passes.hpp"
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
// LowerVPU2VPUIP
//

void vpux::buildLowerIERT2VPUIPPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(createBufferizeFuncAndReturnPass(log));
    pm.addPass(createAddBuffersForNetResults(log));

    pm.addPass(createConvertSWLayers2VPUIPPass(log));
    pm.addPass(createConvertLayers2VPUIPPass(log));

    pm.addPass(createConvertVPUNCEToVPUIPPass(log));
    pm.addPass(createConvertNCEClusterTilingToVPUIPPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// LowerVPUIP2VPUIPRegMapped
//

void vpux::buildLowerVPUIP2VPUIPRegMappedAndELFPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(createConvertVPUIP2VPUIPRegMappedPass(log));
    pm.addPass(createConvert2VPUIPRegMappedAndELFPass(log));
    pm.addPass(vpux::ELF::createRemoveEmptyELFSectionsPass(log));
}

//
// registerConversionPipelines
//

void vpux::registerConversionPipelines() {
    mlir::PassPipelineRegistration<>("lower-IE-to-IERT", "Performs full lowering from the IE Dialect to IERT Dialect",
                                     [](mlir::OpPassManager& pm) {
                                         buildLowerIE2IERTPipeline(pm);
                                     });

    mlir::PassPipelineRegistration<>("lower-VPU-to-VPUIP",
                                     "Performs full lowering from the VPU Dialect to VPUIP Dialect",
                                     [](mlir::OpPassManager& pm) {
                                         buildLowerIERT2VPUIPPipeline(pm);
                                     });

    mlir::PassPipelineRegistration<>(
            "VPUIP-to-VPUIPRegMappedAndELF",
            "Performs full lowering from the VPUIP Dialect to the VPUIPRegMapped and ELF Dialects",
            [](mlir::OpPassManager& pm) {
                buildLowerVPUIP2VPUIPRegMappedAndELFPipeline(pm);
            });
}
