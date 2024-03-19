//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU30XX/dialect/IE/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

void vpux::IE::arch30xx::buildInitialTransformationsPipeline(mlir::OpPassManager& pm, const TransformOptions& options,
                                                             Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createConvertScalarToTensorPass(log));
    pm.addPass(IE::createWeightsDequantizeToFakeQuantizePass(log));
    pm.addPass(IE::createNormalizeL2FusionPass(log));
    pm.addPass(IE::createDecomposeLSTMCellPass(log));
    pm.addPass(IE::createMatMulInputsTo2dPass(log));
    pm.addPass(IE::createRemoveViewLikeOpsChainPass(log));
    pm.addPass(IE::createPropagateOpThroughBatchConcatPass(log));
    pm.addPass(IE::createUnrollFakeQuantizePass(log));
    pm.addPass(IE::createUnrollMatMulPass(log));
    pm.addPass(IE::createConvertMatMulToConvPass(log));
    pm.addPass(IE::createConvertConvBackpropDataToTransposedConvPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableConvertFCToConv) {
        pm.addPass(IE::createConvertFCToConvPass(log));
    }
}

void vpux::IE::arch30xx::buildOptimizeActivationsPipeline(mlir::OpPassManager& pm,
                                                          const OptimizeActivationsOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::arch30xx::createInsertIdentityPoolBeforeOpPass(log));
    pm.addPass(IE::createFuseActivationOpsPass(options.enableFuseClampOperations, log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

void vpux::IE::arch30xx::buildMemPermuteProcessingPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createMovePermutePostEltwisePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createLegalizeNDMemPermutePass(log));
    pm.addPass(IE::createPropagateMemPermuteBeforeOpPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createPropagateMemPermuteThroughAddPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createUniquifyOpsPass(log));
}

//
// LowPrecision
//

void vpux::IE::arch30xx::buildLowPrecisionPipeline(mlir::OpPassManager& pm, const LowPrecisionOptions& options,
                                                   Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createFoldReLUBeforeFQPass(log));
    pm.addPass(IE::createOptimizeUnalignedQDQSeqPass(log));
    pm.addPass(IE::createSwapFakeQuantWithReshapeAndStridedSlicePass(log));
    pm.addPass(IE::createSwapConvertWithTransposeReshapePass(log));
    pm.addPass(IE::createHandleFakeQuantHasNegativeScalesPass(log));
    if (options.enableAlignScales) {
        pm.addPass(IE::createAlignScalesPass(isOptionEnabled(options.enableSEPtrsOperations), log));
    }
    if (options.enableAdjustNonZeroFakeQuant) {
        pm.addPass(IE::createAdjustNonZeroFakeQuantPass(log));
    }
    pm.addPass(IE::createSplitFakeQuantPass(log));
    pm.addPass(IE::createFuseConvertWithQuantizePass(log));
    if (options.enablePropagateQuantDequant) {
        pm.addPass(mlir::createCanonicalizerPass(grc));
        pm.addPass(IE::createPropagateQuantizeDequantizePass(isOptionEnabled(options.enableSEPtrsOperations), log));
    }
    if (options.enableSwapTransposeWithFQ) {
        pm.addPass(IE::createSwapTransposeWithFQPass(log));
    }
    pm.addPass(IE::createPropagateFqThroughConcatPass(log));
    pm.addPass(IE::createConvertWeightsToU8Pass(log));
    pm.addPass(IE::createFuseQuantizedOpsPass(
            /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
            /*enableSEPTransposedConv=*/isOptionEnabled(options.enableSEPTransposedConv), log));
    pm.addPass(IE::arch30xx::createConvertToMixedPrecision(log));
    if (options.enableQuantDequantRemoval) {
        pm.addPass(IE::createRemoveQuantDequantSeqPass(log));
    }
    pm.addPass(IE::createConvertWeightsToU8Pass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createDequantizeConstPass(log));
    pm.addPass(IE::createConvertQuantizeOpsToNceOpsPass(log));
    pm.addPass(IE::createMergeFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// DefaultHWPipeline
//

void vpux::IE::arch30xx::buildDefaultHWPipeline(mlir::OpPassManager& pm, const IE::arch30xx::DefaultHWOptions& options,
                                                Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // Level 3 : Topology
    if (options.logOpOptimizations) {
        pm.addPass(IE::createLogOpOptimizationsPass());
    }
    pm.addPass(IE::createWeightsDequantizeToFakeQuantizePass(log));
    pm.addPass(IE::createNormalizeL2FusionPass(log));
    pm.addPass(IE::createMatMulInputsTo2dPass(log));
    pm.addPass(IE::createConvertMatMulToConvPass(log));
    pm.addPass(IE::createConvertConvBackpropDataToTransposedConvPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    if (options.enableConvertFCToConv) {
        pm.addPass(IE::createConvertFCToConvPass(log));
    }
    pm.addPass(IE::createConvertSquaredDiffToSubAndPowerPass(log));

    IE::buildOperationConversionPipeline(pm, log);

    pm.addPass(IE::createConvertNceOpsTo4DPass(log));
    if (options.enableHandleLargeKernel) {
        pm.addPass(IE::createAdjustMaxPoolInputShapePass(log));
        pm.addPass(IE::createHandleLargeKernelsPass(log));
    }
    pm.addPass(IE::createHandleExcludePadForAvgPoolPass(log));
    if (options.enableConvertAvgPoolToDWConv) {
        pm.addPass(IE::createConvertAvgPoolToDWConvPass(log));
    }

    pm.addPass(IE::createAdaptShapesForScaleShiftPass(log));
    pm.addPass(IE::createResolveStridedSlicePass(log));
    pm.addPass(IE::createSwapTransposeConcatPass(log));
    pm.addPass(IE::createConvertSplitConcatToTransposePass(log));
    pm.addPass(IE::createConvertShapeTo4DPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createConvertToSpatialInterpolationPass(log));
    pm.addPass(IE::createSwapOperationsPass(
            isOptionEnabled(options.enableSEPtrsOperations) || isOptionEnabled(options.enableSEPTransposedConv), log));
    pm.addPass(IE::createSwapPadLayerPass(log));
    pm.addPass(IE::createConvertSubtractToAddPass(log));
    pm.addPass(IE::createConvertToScaleShiftPass(log));
    pm.addPass(IE::createBroadcastInputForAddPass(log));
    pm.addPass(IE::createConvertGRNToNormalizeL2Pass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    // E#79878: Solve eltwise single layer test failure.
    // SwapOperations pass may generate non-4D AddOp.
    // If AddOp appears here means that it cannot be fused into NCE task.
    // So convert it's shape to 4D and then convert this AddOp to ScaleShift.
    pm.addPass(IE::createConvertShapeTo4DPass(log));
    pm.addPass(IE::createConvertToScaleShiftPass(log));
    pm.addPass(IE::createResolveScatterUpdateByTransposePass(log));
    pm.addPass(IE::createConvertGroupConvToConvPass(log));
    pm.addPass(IE::createSwapOperationsPass(
            isOptionEnabled(options.enableSEPtrsOperations) || isOptionEnabled(options.enableSEPTransposedConv), log));

    IE::buildAdjustForVPUPipeline(pm, IE::AdjustForVPUOptions(options), log);

    pm.addPass(IE::arch30xx::createConvertTile2PerAxisTilePass(log));

    if (options.enableSwapTransposeWithFQ) {
        pm.addPass(IE::createSwapTransposeWithFQPass(log));
    }
    if (options.enableSplitConvWithMultipleFQ) {
        pm.addPass(IE::createSplitConvWithMultipleFQPass(log));
    }
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableHandleLargeStrides) {
        pm.addPass(IE::createHandleLargeStridesPass(log));
    }
    if (options.enableHandleAsymmetricStrides) {
        pm.addPass(IE::createHandleAsymmetricStridesPass(log));
    }
    if (options.enableHandleLargePads) {
        pm.addPass(IE::createHandleLargePadsPass(log));
    }
    pm.addPass(IE::createConvertGroupConvToConvPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    if (options.enableOptimizeScaleShiftToDWConv) {
        IE::buildScaleShiftProcessingPipeline(pm, log);
    }

    pm.addPass(IE::createFuseActivationOpsPass(options.enableFuseClampOperations, log));
    pm.addPass(IE::createConvertStridedSlice2ConvPass(log));
    if (options.enableLowPrecision) {
        IE::arch30xx::buildLowPrecisionPipeline(pm, IE::LowPrecisionOptions(options), log);
        pm.addPass(IE::createConvertShapeTo4DPass(log));
        pm.addPass(IE::createSwapViewOpAndClampPass(log));
    }
    IE::arch30xx::buildOptimizeActivationsPipeline(pm, IE::OptimizeActivationsOptions(options), log);

    if (options.enableSEPtrsOperations && options.enableSplitBilinerIntoHAndW) {
        pm.addPass(IE::createSplitBilinerIntoHAndWPass(log));
    }

    if (options.enableBilinearInterpolateOnDPU) {
        pm.addPass(IE::arch30xx::createMapBilinearInterpolateOnDPUPass(isOptionEnabled(options.enableSEPtrsOperations),
                                                                       log));
    }

    pm.addPass(IE::createConvertBatchedLayerTo1NPass(log));
    pm.addPass(IE::arch30xx::createUnrollBatchPass(log));

    if (options.enableUpstreamSlice) {
        pm.addPass(IE::createUpstreamSlicePass(log));
    }
    pm.addPass(IE::createFuseConvWithSlicePass(log));

    pm.addPass(IE::createSwapMVNWithTransposePass(log));

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);
    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));

    if (options.enableExpandActivationChannels) {
        pm.addPass(IE::arch30xx::createExpandActivationChannelsPass(
                /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
                /*seTransposedConvEnabled=*/isOptionEnabled(options.enableSEPTransposedConv), log));
        pm.addPass(mlir::createCanonicalizerPass(grc));

        if (options.enableOptimizeSliceExpand) {
            pm.addPass(IE::arch30xx::createOptimizeSliceExpandPass(log));
        }

        pm.addPass(IE::createAdjustConvolutionInputShapePass(log));
        pm.addPass(IE::createAdjustInputShapeForEltwisePass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
        if (options.enableOptimizeSliceExpand) {
            pm.addPass(IE::arch30xx::createOptimizeSliceExpandPass(log));
        }

        if (options.enableOptimizeReorders) {
            pm.addPass(IE::createOptimizeReordersPass(
                    /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
                    /*seTransposedConvEnabled=*/isOptionEnabled(options.enableSEPTransposedConv), log));
            pm.addPass(IE::createUniquifyOpsPass(log));
            pm.addPass(IE::createPropagateAffineReshapePass(log));
            pm.addPass(IE::createUniquifyBranchesPass(log));
        }
    }

    pm.addPass(IE::createSwapOperationsPass(
            isOptionEnabled(options.enableSEPtrsOperations) || isOptionEnabled(options.enableSEPTransposedConv), log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createConvertSplitConcatToTransposePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createMovePermutePostEltwisePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createPropagateMemPermuteThroughAddPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createUniquifyOpsPass(log));
    if (options.enableConvertSliceToConvPass) {
        pm.addPass(IE::createConvertSliceToConvPass(log));
    }
    if (options.enableConvertExpandToConvPass) {
        pm.addPass(IE::createConvertExpandToConvPass(log));
    }
    pm.addPass(IE::createOptimizeIdentityPoolPass(log));
    if (options.logOpOptimizations) {
        pm.addPass(IE::createLogOpOptimizationsPass());
    }
}

//
// registerIEPipelines
//

void vpux::IE::arch30xx::registerIEPipelines() {
    mlir::PassPipelineRegistration<IE::arch30xx::DefaultHWOptions>(
            "default-hw-mode-ie", "IE dialect part of Default HW pipeline",
            [](mlir::OpPassManager& pm, const IE::arch30xx::DefaultHWOptions& options) {
                IE::arch30xx::buildDefaultHWPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<TransformOptions>(
            "initial-transformations",
            "[LEGALIZATION] Initial Transformations, convert initial IR operations to another and tries to reduce the "
            "number of op types used in the graph",
            [](mlir::OpPassManager& pm, const TransformOptions& options) {
                IE::arch30xx::buildInitialTransformationsPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<OptimizeActivationsOptions>(
            "optimize-activations", "[OPTIMIZATION] Optimize activations for VPU target",
            [](mlir::OpPassManager& pm, const OptimizeActivationsOptions& options) {
                IE::arch30xx::buildOptimizeActivationsPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<mlir::EmptyPipelineOptions>(
            "mempermute-processing",
            "[OPTIMIZATION] MemPermute processing is responsible for handling data transfromations ops (Transpose, "
            "Reshape etc), transform it to MemPermute and optimize final subgraph to avoid unnesesary data "
            "permutations",
            [](mlir::OpPassManager& pm) {
                IE::arch30xx::buildMemPermuteProcessingPipeline(pm);
            });

    mlir::PassPipelineRegistration<LowPrecisionOptions>(
            "low-precision", "[OPTIMIZATION] Low precision transformations",
            [](mlir::OpPassManager& pm, const LowPrecisionOptions& options) {
                IE::arch30xx::buildLowPrecisionPipeline(pm, options);
            });
}
