//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/pipelines.hpp"
#include "vpux/compiler/VPU37XX/dialect/VPU/passes.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

void vpux::VPU::arch37xx::buildIncrementalPipeline(mlir::OpPassManager& pm, const VPU::TilingOptions& options,
                                                   Logger log) {
    pm.addPass(VPU::createMultiClusterStrategyAssignmentPass(log));

    // manual strategy debug configuration
    StringRef writeStrategyFileLocation = "strategy_out.json";
    StringRef readStrategyFileLocation = "strategy_in.json";

    pm.addPass(VPU::createManualStrategyUtilsPass(options.writeStrategyToJson, writeStrategyFileLocation,
                                                  options.readStrategyFromJson, readStrategyFileLocation, log));

    pm.addPass(VPU::createSplitGRUSequencePass(log));
    pm.addPass(VPU::arch37xx::createDecomposeGatherPass(log));

    VPU::buildTilingPipeline(pm, options, log);

    pm.addPass(VPU::createManualStrategyUtilsPass(options.writeStrategyToJson, writeStrategyFileLocation,
                                                  options.readStrategyFromJson, readStrategyFileLocation, log));

    pm.addPass(VPU::createAdjustTilingForPermuteQuantizePass(log));

    pm.addPass(VPU::createWrapVPUOpsInNCEClusterTilingPass(options.enableExplicitDistributedTensorAttr, log));
}

void vpux::VPU::arch37xx::registerVPUPipelines() {
    mlir::PassPipelineRegistration<VPU::TilingOptions>("incremental-pipeline", "Apply Incremental Pipeline",
                                                       [](mlir::OpPassManager& pm, const VPU::TilingOptions& options) {
                                                           VPU::arch37xx::buildIncrementalPipeline(pm, options);
                                                       });
}
