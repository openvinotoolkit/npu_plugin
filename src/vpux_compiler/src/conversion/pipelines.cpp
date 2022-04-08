//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/dialect/ELF/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/passes.hpp"
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
// LowerIE2VPU
//

void vpux::buildLowerIE2VPUPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(createConvertIEToVPUNCEPass(log));
    pm.addPass(createConvertLayers2VPUPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// LowerVPU2VPUIP
//

void vpux::buildLowerVPU2VPUIPPipeline(mlir::OpPassManager& pm, Logger log) {
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

void vpux::buildLowerVPUIP2ELFPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(createConvertVPUIP2VPUIPRegMappedPass(log));
    pm.addPass(VPUIPRegMapped::createBarrierComputationPass(log));

    pm.addPass(createConvertVPUIPRegMapped2ELFPass(log));
    pm.addPass(ELF::createRemoveEmptyELFSectionsPass(log));
    pm.addPass(ELF::createUpdateELFSectionFlagsPass(log));
}

//
// registerConversionPipelines
//

void vpux::registerConversionPipelines() {
    mlir::PassPipelineRegistration<>("lower-IE-to-IERT", "Performs full lowering from the IE Dialect to IERT Dialect",
                                     [](mlir::OpPassManager& pm) {
                                         buildLowerIE2IERTPipeline(pm);
                                     });

    mlir::PassPipelineRegistration<>("lower-IE-to-VPU", "Performs full lowering from the IE Dialect to VPU Dialect",
                                     [](mlir::OpPassManager& pm) {
                                         buildLowerIE2VPUPipeline(pm);
                                     });

    mlir::PassPipelineRegistration<>("lower-VPU-to-VPUIP",
                                     "Performs full lowering from the VPU Dialect to VPUIP Dialect",
                                     [](mlir::OpPassManager& pm) {
                                         buildLowerVPU2VPUIPPipeline(pm);
                                     });

    mlir::PassPipelineRegistration<>(
            "lower-VPUIP-to-ELF",
            "Performs full lowering from the VPUIP Dialect to the VPUIPRegMapped and ELF Dialects",
            [](mlir::OpPassManager& pm) {
                buildLowerVPUIP2ELFPipeline(pm);
            });
}
