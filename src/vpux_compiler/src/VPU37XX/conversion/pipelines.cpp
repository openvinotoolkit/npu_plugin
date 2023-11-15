//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU37XX/conversion.hpp"

#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/dialect/ELF/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// LowerIE2VPU
//

void vpux::arch37xx::buildLowerIE2VPUPipeline37XX(mlir::OpPassManager& pm,
                                                  const vpux::arch37xx::PermuteQuantOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(vpux::arch37xx::createConvertIEToVPUNCEPass(options.useNCEPermute, log));
    pm.addPass(createConvertLayers2VPUPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// LowerVPU2VPUIPSWKernel
//

void vpux::arch37xx::buildLowerVPU2VPUIP37XXPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(createBufferizeFuncAndReturnPass(log));
    pm.addPass(createAddBuffersForNetResults(log));

    pm.addPass(createConvertSWLayers2VPUIPSWKernelPass(log));
    pm.addPass(createConvertLayers2VPUIPPass(log));

    pm.addPass(createConvertVPUNCEToVPUIPPass(log));
    pm.addPass(createConvertNCEClusterTilingToVPUIPPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// LowerVPUIP2VPUMI37XXAndELF
//

void vpux::arch37xx::buildLowerVPUIP2ELFPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(createConvertVPUIP2VPUMI37XXPass(log));
    pm.addPass(VPUMI37XX::createBarrierComputationPass(log));

    pm.addPass(createConvertVPUMI37XX2ELFPass(log));
    pm.addPass(ELF::createRemoveEmptyELFSectionsPass(log));
    pm.addPass(ELF::createUpdateELFSectionFlagsPass(log));
}

//
// registerConversionPipelines
//

void vpux::arch37xx::registerConversionPipeline37XX() {
    mlir::PassPipelineRegistration<PermuteQuantOptions>(
            "lower-IE-to-VPU", "Performs full lowering from the IE Dialect to VPU Dialect",
            [](mlir::OpPassManager& pm, const PermuteQuantOptions& options) {
                vpux::arch37xx::buildLowerIE2VPUPipeline37XX(pm, options);
            });

    mlir::PassPipelineRegistration<>(
            "lower-VPU-to-VPUIP",
            "Performs full lowering from the VPU Dialect to VPUIP Dialect, SW operations are converted to SWKernelOp",
            [](mlir::OpPassManager& pm) {
                vpux::arch37xx::buildLowerVPU2VPUIP37XXPipeline(pm);
            });

    mlir::PassPipelineRegistration<>("lower-VPUIP-to-ELF",
                                     "Performs full lowering from the VPUIP Dialect to the VPUMI37XX and ELF Dialects",
                                     [](mlir::OpPassManager& pm) {
                                         vpux::arch37xx::buildLowerVPUIP2ELFPipeline(pm);
                                     });
}
