//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// AdjustPrecision
//

void vpux::IE::buildAdjustPrecisionPipeline(mlir::OpPassManager& pm, const AdjustPrecisionOptions& options,
                                            Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createConvertPrecisionToFP16Pass(log));
    pm.addPass(IE::createConvertPrecisionToI32Pass(log));
    if (options.enableUseUserPrecision) {
        pm.addPass(IE::createUseUserPrecisionPass(log));
    }
    pm.addPass(IE::createAdjustSoftwareOpsPrecisionPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// AdjustLayout
//

void vpux::IE::buildAdjustLayoutPipeline(mlir::OpPassManager& pm, const AdjustLayoutOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    if (options.enableUseUserLayout) {
        pm.addPass(IE::createUseUserLayout(log));
    }

    if (options.enableForceZMajorConcat) {
        pm.addPass(IE::createInsertReorderBetweenLayerAndConcatPass(log));
    }

    pm.addPass(IE::createPropagateAffineReshapePass(log));
    pm.addPass(IE::createUniquifyBranchesPass(log));
    pm.addPass(IE::createSwapTransposeConcatPass(log));
    pm.addPass(IE::createTransposeToPermuteCastPass(log));
    pm.addPass(IE::createAdjustLayoutsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableOptimizeReorders) {
        pm.addPass(IE::createOptimizeReordersPass(log));
        pm.addPass(IE::createUniquifyOpsPass(log));
        pm.addPass(IE::createUniquifyBranchesPass(log));
        pm.addPass(IE::createPropagateReorderToNCEPass(log));
        pm.addPass(IE::createFuseReordersPass(log));
    }
}

//
// AdjustForVPU
//

void vpux::IE::buildAdjustForVPUPipeline(mlir::OpPassManager& pm, const AdjustForVPUOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createConvertTile2PerAxisTilePass(log));
    pm.addPass(IE::createConvertConv1DToConv2DPass(log));
    pm.addPass(IE::createLegalizeDilatedConvolutionPass(log));
    pm.addPass(IE::createPerAxisFQConcatPass(log));
    pm.addPass(IE::createConvertPaddingsToFloorModePass(log));
    pm.addPass(IE::createConvertShuffleChannelsPass(log));
    pm.addPass(IE::createConvertNearestToStridedConcatPass(log));
    pm.addPass(IE::createConvertBilinearToStridedConcatAndConvPass(log));
    pm.addPass(IE::createConvertDeconv2DToConv2DPass(log));
    pm.addPass(IE::createConvertUpsamplingToStridedConcatPass(log));
    pm.addPass(IE::createFusePadOpsPass(log));
    pm.addPass(IE::createConvertPadToConcatPass(log));
    pm.addPass(IE::createConvertDepth2SpaceLayerPass(log));
    pm.addPass(IE::createConvertSpace2DepthLayerPass(log));
    pm.addPass(IE::createConvertGatherToSlicePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableSwapConcatWithEltwise) {
        pm.addPass(IE::createSwapConcatWithEltwisePass(log));
    }
    pm.addPass(IE::createSwapMaxPoolWithActivation(log));
    pm.addPass(IE::createFusePostOpsPass(log));
    pm.addPass(IE::createInsertMaxpoolToConcatActivationPass(log));
    pm.addPass(IE::createOptimizeConcatSlicePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// LowPrecision
//

void vpux::IE::buildLowPrecisionPipeline(mlir::OpPassManager& pm, const LowPrecisionOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createOptimizeUnalignedQDQSeqPass(log));
    pm.addPass(IE::createSwapFakeQuantReshapePass(log));
    pm.addPass(IE::createSwapConvertWithTransposeReshapePass(log));
    if (options.enableAlignScales) {
        pm.addPass(IE::createAlignScalesPass(log));
    }
    pm.addPass(IE::createSplitFakeQuantPass(log));
    pm.addPass(IE::createFuseConvertWithQuantizePass(log));
    if (options.enablePropagateQuantDequant) {
        pm.addPass(mlir::createCanonicalizerPass(grc));
        pm.addPass(IE::createPropagateQuantizeDequantizePass(log));
    }
    if (options.enableSwapTransposeWithFQ) {
        pm.addPass(IE::createSwapTransposeWithFQPass(log));
    }
    pm.addPass(IE::createPropagateFqThroughConcatPass(log));
    pm.addPass(IE::createFuseQuantizedOpsPass(log));
    pm.addPass(IE::createConvertToMixedPrecision(options.enableFP16ToU8MixedMode, log));
    if (options.enableQuantDequantRemoval) {
        pm.addPass(IE::createRemoveQuantDequantSeqPass(log));
    }
    pm.addPass(IE::createConvertWeightsToU8Pass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createDequantizeConstPass(log));

    if (options.forceHostInputQuantization) {
        // At this point we have IE::QuantizeOps that are not yet converted
        // to NCE ops. We can remove them now in order to perform input
        // quantization on the host side.
        pm.addPass(IE::createForceHostQuantizationPass(log));
    }

    pm.addPass(IE::createConvertQuantizeOpsToNceOpsPass(log));
    pm.addPass(IE::createDeletePerAxisQuantizationPass(log));
    pm.addPass(IE::createMergeFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// registerIEPipelines
//

void vpux::IE::registerIEPipelines() {
    mlir::PassPipelineRegistration<AdjustPrecisionOptions>(
            "adjust-precision", "Adjust IR precision for VPU target",
            [](mlir::OpPassManager& pm, const AdjustPrecisionOptions& options) {
                IE::buildAdjustPrecisionPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<AdjustLayoutOptions>(
            "adjust-layout", "Adjust IR layout for VPU target",
            [](mlir::OpPassManager& pm, const AdjustLayoutOptions& options) {
                IE::buildAdjustLayoutPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<AdjustForVPUOptions>(
            "adjust-for-vpu", "Adjust IE Dialect IR for VPU target",
            [](mlir::OpPassManager& pm, const AdjustForVPUOptions& options) {
                IE::buildAdjustForVPUPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<LowPrecisionOptions>(
            "low-precision", "Low precision transformations",
            [](mlir::OpPassManager& pm, const LowPrecisionOptions& options) {
                IE::buildLowPrecisionPipeline(pm, options);
            });
}
