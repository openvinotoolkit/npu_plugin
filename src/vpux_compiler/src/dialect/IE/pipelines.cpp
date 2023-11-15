//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/pipelines.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// AdjustPrecision
//

void vpux::IE::buildAdjustPrecisionPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createConvertPrecisionToFP16Pass(log));
    pm.addPass(IE::createConvertPrecisionToI32Pass(log));
    pm.addPass(IE::createUseUserPrecisionPass(log));
    pm.addPass(IE::createAdjustSoftwareOpsPrecisionPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// AdjustLayout
//

void vpux::IE::buildAdjustLayoutPipeline(mlir::OpPassManager& pm, const AdjustLayoutOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    if (options.enableForceZMajorConcat) {
        pm.addPass(IE::createInsertReorderBetweenLayerAndConcatPass(log));
    }

    pm.addPass(IE::createPropagateAffineReshapePass(log));
    pm.addPass(IE::createPropagateTransposePass(log));
    pm.addPass(IE::createUniquifyBranchesPass(log));
    pm.addPass(IE::createSwapTransposeConcatPass(log));
    pm.addPass(IE::createTransposeToPermuteCastPass(log));
    pm.addPass(IE::createAdjustLayoutsPass(/*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations), log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableOptimizeReorders) {
        pm.addPass(
                IE::createOptimizeReordersPass(/*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations), log));
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

    pm.addPass(IE::createLegalizeDilatedConvolutionPass(log));
    pm.addPass(IE::createPerAxisFQConcatPass(log));
    pm.addPass(IE::createConvertPaddingsToFloorModePass(log));
    pm.addPass(IE::createConvertShuffleChannelsPass(log));
    pm.addPass(IE::createConvertNearestToBroadCastOrStridedConcatPass(
            /*interpolateAsSEOp=*/isOptionEnabled(options.enableSEPtrsOperations), log));
    pm.addPass(IE::createConvertBilinearToStridedConcatAndConvPass(
            /*interpolateAsSEOp=*/isOptionEnabled(options.enableSEPtrsOperations), log));
    pm.addPass(IE::createConvertBroadcastToTilePass(log));
    pm.addPass(IE::createConvertScatterNDUpdateToStridedConcatPass(log));
    pm.addPass(IE::createConvertDeconv2DToConv2DPass(log));
    pm.addPass(IE::createConvertGroupDeconvToGroupConvPass(log));
    pm.addPass(IE::createConvertGroupConvToConvPass(log));
    pm.addPass(IE::createConvertUpsamplingToStridedConcatPass(log));
    pm.addPass(IE::createConvertReflectPadToSliceAndConcatPass(log));
    pm.addPass(IE::createFusePadOpsPass(log));
    pm.addPass(IE::createConvertPadToConcatPass(log));
    pm.addPass(IE::createConvertDepth2SpaceLayerPass(log));
    pm.addPass(IE::createConvertSpace2DepthLayerPass(log));
    pm.addPass(IE::createConvertGatherToSlicePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createSwapMaxPoolWithActivation(log));
    pm.addPass(IE::createFusePostOpsPass(log));
    pm.addPass(IE::createOptimizeConcatSlicePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// LowPrecision
//

void vpux::IE::buildLowPrecisionPipeline(mlir::OpPassManager& pm, const LowPrecisionOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createFoldReLUBeforeFQPass(log));
    pm.addPass(IE::createOptimizeUnalignedQDQSeqPass(log));
    pm.addPass(IE::createSwapFakeQuantWithReshapeAndStridedSlicePass(log));
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
    pm.addPass(IE::createFuseQuantizedOpsPass(
            /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations), log));
    pm.addPass(IE::createConvertToMixedPrecision(options.enableFP16ToU8MixedMode, log));
    if (options.enableQuantDequantRemoval) {
        pm.addPass(IE::createRemoveQuantDequantSeqPass(log));
    }
    pm.addPass(IE::createConvertWeightsToU8Pass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createDequantizeConstPass(log));
    pm.addPass(IE::createConvertQuantizeOpsToNceOpsPass(log));
    pm.addPass(IE::createMergeFakeQuantPass(log));
    pm.addPass(IE::createSwapQuantCastAndClampPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

void vpux::IE::buildInitialTransformationsPipeline(mlir::OpPassManager& pm, const TransformOptions& options,
                                                   Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createConvertScalarToTensorPass(log));
    pm.addPass(IE::createNormalizeL2FusionPass(log));
    pm.addPass(IE::createDecomposeLSTMCellPass(log));
    pm.addPass(IE::createMatMulInputsTo2dPass(log));
    pm.addPass(IE::createPropagateOpThroughBatchConcatPass(log));
    pm.addPass(IE::createConvertMatMulToConvPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableConvertFCToConv) {
        pm.addPass(IE::createConvertFCToConvPass(log));
    }
}

void vpux::IE::buildScaleShiftProcessingPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createAdjustScaleShiftForDWConvPass(log));
    pm.addPass(IE::createConvertBroadcastToTilePass(log));
    pm.addPass(IE::createConvertScaleShiftToDWPass(log));

    pm.addPass(mlir::createCanonicalizerPass(grc));
}

void vpux::IE::buildOperationConversionPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();
    pm.addPass(IE::createConvertExtractImagePatchesPass(log));
    pm.addPass(IE::createConvertReduceSumToConvPass(log));
    pm.addPass(IE::createConvertReduceToPoolingPass(log));
    pm.addPass(IE::createConvertSquaredDiffToSubAndPowerPass(log));
    pm.addPass(IE::createConvertPowerToMultPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// registerIEPipelines
//

void vpux::IE::registerIEPipelines() {
    mlir::PassPipelineRegistration<>("adjust-precision", "[LEGALIZATION] Adjust IR precision for VPU target",
                                     [](mlir::OpPassManager& pm) {
                                         IE::buildAdjustPrecisionPipeline(pm);
                                     });

    mlir::PassPipelineRegistration<AdjustLayoutOptions>(
            "adjust-layout", "[LEGALIZATION] Adjust IR layout for VPU target",
            [](mlir::OpPassManager& pm, const AdjustLayoutOptions& options) {
                IE::buildAdjustLayoutPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<AdjustForVPUOptions>(
            "adjust-for-vpu", "[LEGALIZATION] Adjust IE Dialect IR for VPU target",
            [](mlir::OpPassManager& pm, const AdjustForVPUOptions& options) {
                IE::buildAdjustForVPUPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<LowPrecisionOptions>(
            "low-precision", "[OPTIMIZATION] Low precision transformations",
            [](mlir::OpPassManager& pm, const LowPrecisionOptions& options) {
                IE::buildLowPrecisionPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<TransformOptions>(
            "initial-transformations",
            "[LEGALIZATION] Initial Transformations, convert initial IR operations to another and tries to reduce the "
            "number of op types used in the graph",
            [](mlir::OpPassManager& pm, const TransformOptions& options) {
                IE::buildInitialTransformationsPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<mlir::EmptyPipelineOptions>(
            "scaleshift-processing",
            "[OPTIMIZATION] scaleshift processing is responsible for handling scaleshift ops, transform it to"
            "depthwise convolution and optimize final subgraph to run more efficiently",
            [](mlir::OpPassManager& pm) {
                IE::buildScaleShiftProcessingPipeline(pm);
            });

    mlir::PassPipelineRegistration<mlir::EmptyPipelineOptions>(
            "operation-conversion",
            "[OPTIMIZATION] Operation Coversion pipeline is responsible for changing type of existing operations. Main "
            "purpose is reducing subset of ops"
            "which using in our graph for improve pattern matching of next passes ",
            [](mlir::OpPassManager& pm) {
                IE::buildOperationConversionPipeline(pm);
            });
}
