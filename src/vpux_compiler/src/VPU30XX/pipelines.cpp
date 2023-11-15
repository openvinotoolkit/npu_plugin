//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU30XX/pipelines.hpp"

#include "vpux/compiler/VPU30XX/conversion.hpp"
#include "vpux/compiler/VPU30XX/dialect/IE/passes.hpp"
#include "vpux/compiler/VPU30XX/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/dialect/EMU/passes.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPURT/passes.hpp"
#include "vpux/compiler/dialect/const/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "vpux/compiler/VPU30XX/dialect/VPU/passes.hpp"

using namespace vpux;

//
// ReferenceSWMode
//

void vpux::buildReferenceSWModePipeline(mlir::OpPassManager& pm, const ReferenceSWOptions30XX& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // Level 3 : Topology

    pm.addPass(IE::createConvertScalarToTensorPass(log));
    pm.addPass(IE::createNormalizeL2FusionPass(log));
    pm.addPass(IE::createMatMulInputsTo2dPass(log));
    pm.addPass(IE::createConvertMatMulToConvPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustPrecisionPipeline(pm, log);

    pm.addPass(IE::createConvertSquaredDiffToSubAndPowerPass(log));
    pm.addPass(IE::createResolveStridedSlicePass(log));
    pm.addPass(IE::createConvertNceOpsTo4DPass(log));
    pm.addPass(IE::createConvertShapeTo4DPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createConvertGRNToNormalizeL2Pass(log));
    IE::buildAdjustForVPUPipeline(pm, IE::AdjustForVPUOptions(options), log);
    pm.addPass(IE::arch30xx::createConvertTile2PerAxisTilePass(log));

    pm.addPass(IE::createSplitFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createDequantizeConstPass(log));
    if (options.enableMergeFakeQuant) {
        pm.addPass(IE::createMergeFakeQuantPass(log));
    }
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);
    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));

    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // Lowering to VPU
    pm.addPass(createConvertLayers2VPUPass(log));

    pm.addPass(VPU::createTilingStrategyAssignmentPass(/*enablePrefetchTiling=*/false, log));
    pm.addPass(VPU::createApplyTilingPass(log));

    // Lowering to VPUIP
    pm.addPass(createBufferizeFuncAndReturnPass(log));
    pm.addPass(createAddBuffersForNetResults(log));

    pm.addPass(createConvertSWLayers2VPUIPUPAPass(log));
    pm.addPass(createConvertLayers2VPUIPPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // Lowering to VPUIP
    pm.addPass(createConvertLayers2VPUIPPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // Level 2 : Abstract RunTime

    pm.addPass(VPUIP::createSetMemorySpacePass(getMemKind<VPU::MemoryKind::DDR>, log));

    pm.addPass(VPUIP::createCopyOpTilingPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    VPUIP::buildAsyncSchedulingPipeline(pm, log);

    pm.addPass(VPUIP::createStaticAllocationPass(getMemKind<VPU::MemoryKind::CMX_NN>, log));
    pm.addPass(VPUIP::createStaticAllocationPass(getMemKind<VPU::MemoryKind::DDR>, log));
    pm.addPass(VPUIP::createLinearizationPass(log));
    pm.addPass(VPUIP::createOptimizeAsyncDepsPass(log));

    pm.addPass(VPUIP::createBreakDataFlowPass(log));

    VPUIP::buildHardwareAdaptationPipeline(pm, log);

    // Level 1 : VPU RunTime

    if (options.enableProfiling) {
        if (options.enableSWProfiling) {
            pm.addPass(VPUIP::createUPAProfilingPass(log));
        }
        pm.addPass(VPUIP::createGroupProfilingBuffersPass(log));
        pm.addPass(createMoveDeclarationsToTopPass(log));
    }

    pm.addPass(VPUIP::createConvertFuncArgsToDeclarationsPass(log));
    pm.addPass(VPURT::createAssignPhysicalBarriersPass(log));
    pm.addPass(VPURT::createBarrierSimulationPass(log));
    pm.addPass(VPUIP::createDumpStatisticsOfTaskOpsPass(false, log));
    pm.addPass(Const::createConstantFoldingPass());
}

//
// ReferenceHWMode
//

void vpux::buildReferenceHWModePipeline(mlir::OpPassManager& pm, const ReferenceHWOptions30XX& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // Level 3 : Topology

    pm.addPass(IE::createConvertScalarToTensorPass(log));
    pm.addPass(IE::createNormalizeL2FusionPass(log));
    pm.addPass(IE::createMatMulInputsTo2dPass(log));
    pm.addPass(IE::createConvertMatMulToConvPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustPrecisionPipeline(pm, log);

    if (options.enableConvertFCToConv) {
        pm.addPass(IE::createConvertFCToConvPass(log));
    }

    IE::buildOperationConversionPipeline(pm, log);

    pm.addPass(IE::createConvertNceOpsTo4DPass(log));
    if (options.enableHandleLargeKernel) {
        pm.addPass(IE::createHandleLargeKernelsPass(log));
    }
    pm.addPass(IE::createHandleExcludePadForAvgPoolPass(log));
    if (options.enableConvertAvgPoolToDWConv) {
        pm.addPass(IE::createConvertAvgPoolToDWConvPass(log));
    }

    pm.addPass(IE::createAdaptShapesForScaleShiftPass(log));
    pm.addPass(IE::createResolveStridedSlicePass(log));
    pm.addPass(IE::createConvertShapeTo4DPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createSwapOperationsPass(log));
    pm.addPass(IE::createSwapPadLayerPass(log));
    pm.addPass(IE::createConvertSubtractToAddPass(log));
    pm.addPass(IE::createConvertToScaleShiftPass(log));
    pm.addPass(IE::createConvertGRNToNormalizeL2Pass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createResolveScatterUpdateByTransposePass(log));
    pm.addPass(IE::createConvertGroupConvToConvPass(log));
    pm.addPass(IE::createSwapOperationsPass(log));

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
    pm.addPass(mlir::createCanonicalizerPass(grc));
    if (options.enableOptimizeScaleShiftToDWConv) {
        IE::buildScaleShiftProcessingPipeline(pm, log);
    }

    pm.addPass(IE::createFusePostOpsPass(log));
    if (options.enableLowPrecision) {
        IE::buildLowPrecisionPipeline(pm, IE::LowPrecisionOptions(options), log);
        pm.addPass(IE::createConvertShapeTo4DPass(log));
    }
    pm.addPass(IE::createConvertStridedSlice2DWConvPass(log));
    IE::arch30xx::buildOptimizeActivationsPipeline(pm, log);

    if (options.enableBilinearInterpolateOnDPU) {
        pm.addPass(IE::arch30xx::createMapBilinearInterpolateOnDPUPass(isOptionEnabled(options.enableSEPtrsOperations),
                                                                       log));
    }

    pm.addPass(IE::createConvertBatchedConvTo1NPass(log));
    pm.addPass(IE::arch30xx::createUnrollBatchPass(log));

    if (options.enableUpstreamSlice) {
        pm.addPass(IE::createUpstreamSlicePass(log));
    }

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);
    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));

    if (options.enableExpandActivationChannels) {
        pm.addPass(IE::arch30xx::createExpandActivationChannelsPass(
                /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations), log));
        pm.addPass(mlir::createCanonicalizerPass(grc));

        if (options.enableOptimizeSliceExpand) {
            pm.addPass(IE::arch30xx::createOptimizeSliceExpandPass(log));
        }

        pm.addPass(IE::createAdjustInputShapeForEltwisePass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
        if (options.enableOptimizeSliceExpand) {
            pm.addPass(IE::arch30xx::createOptimizeSliceExpandPass(log));
        }

        if (options.enableOptimizeReorders) {
            pm.addPass(IE::createOptimizeReordersPass(/*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
                                                      log));
            pm.addPass(IE::createUniquifyOpsPass(log));
            pm.addPass(IE::createPropagateAffineReshapePass(log));
            pm.addPass(IE::createUniquifyBranchesPass(log));
        }
    }

    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(IE::createMovePermutePostEltwisePass(log));

    // Lowering to VPU
    vpux::arch30xx::buildLowerIE2VPUPipeline30XX(pm, log);

    pm.addPass(VPU::createResolvePWLPostOpsPass(log));

    if (options.enableSEPtrsOperations) {
        pm.addPass(VPU::createLowerOpsToSENCEPass(log));
    }
    if (options.enableWeightsSparsity) {
        VPU::buildWeightsSparsityPipeline(pm, VPU::WeightsSparsityOptions(options), log);
    }

    pm.addPass(VPU::createEnsureNCEOpsSizeRequirementsPass(log));
    if (options.enableInPlaceEltwise) {
        pm.addPass(VPU::createDetectInPlaceEltwisePass(log));
    }

    if (options.enableSMPipeline) {
        VPU::buildSMPipeline(pm, VPU::TilingOptions(options), log);
    } else {
        VPU::arch30xx::buildIncrementalPipeline(pm, VPU::TilingOptions(options), log);
    }

    pm.addPass(VPU::createOptimizeConcatPass(log));
    pm.addPass(VPU::createAdjustMemorySpacePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPU::createCMXConcatPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPU::createSplitNCEOpsOntoWorkloadsPass(log));
    pm.addPass(VPU::createResolveEltwiseWithZTiledWorkloadsPass(log));

    // Lowering to VPUIP
    vpux::arch30xx::buildLowerVPU2VPUIP30XXPipeline(pm, log);
    pm.addPass(VPUIP::createConvertExpandPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPUIP::createConvertEltwiseToInPlacePass(log));

    // Level 2 : Abstract RunTime

    pm.addPass(VPUIP::createSetMemorySpacePass(getMemKind<VPU::MemoryKind::DDR>, log));

    if (options.enableOptimizeCopies) {
        pm.addPass(VPUIP::createMovePureViewOpBeforeCopyPass(log));
        pm.addPass(VPUIP::createOptimizeCopiesPass(log));
        pm.addPass(VPUIP::createCopyOpHoistingPass(log));
        pm.addPass(VPUIP::createOptimizeParallelCopiesPass(options.enableOptimizeConstCopies, log));
    }

    pm.addPass(VPUIP::createConvertToDMAPass(log));
    pm.addPass(VPUIP::createCopyOpTilingPass(log));

    if (options.enableSEPtrsOperations) {
        pm.addPass(VPUIP::createMoveSubViewBeforeSparseBufferPass(log));
        pm.addPass(VPUIP::createComputeSEBasePtrsPass(log));
        pm.addPass(VPUIP::createConvertSETablesToConstantsPass(log));
    }
    if (options.enableWeightsSparsity) {
        pm.addPass(VPUIP::createPropagateCompressionSchemePass(log));
    }
    if (options.enableWeightsSparsity) {
        pm.addPass(VPUIP::createUngroupSparseBuffersPass(log));
    }

    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableConstantFusion) {
        pm.addPass(VPUIP::createFuseConstantsPass(log));
    }

    if (options.enableProfiling && options.enableDPUProfiling) {
        pm.addPass(VPUIP::createDPUProfilingPass(getMemKind<VPU::MemoryKind::CMX_NN>, log));
    }
    if (options.enableProfiling && options.enableSWProfiling) {
        pm.addPass(VPUIP::createActShaveProfilingPass(getMemKind<VPU::MemoryKind::CMX_NN>, log));
    }

    VPUIP::buildAsyncSchedulingPipeline(pm, log);

    if (options.enableProfiling && options.enableDMAProfiling) {
        pm.addPass(VPUIP::createDMATaskProfilingReserveMemPass(log));
    }

    pm.addPass(VPUIP::createCalculateAsyncRegionCycleCostPass(log));

    pm.addPass(VPUIP::createFeasibleAllocationPass(getMemKind<VPU::MemoryKind::CMX_NN>,
                                                   getMemKind<VPU::MemoryKind::DDR>, log));

    pm.addPass(VPUIP::createMaximizeUPACyclesPass(log));

    if (options.enableGroupAsyncExecuteOps) {
        pm.addPass(VPUIP::createGroupAsyncExecuteOpsPass(log));
    }

    pm.addPass(VPUIP::createStaticAllocationPass(getMemKind<VPU::MemoryKind::DDR>, log));
    pm.addPass(VPUIP::createOptimizeAsyncDepsPass(log));
    pm.addPass(VPUIP::createBreakDataFlowPass(log));

    if (options.enableConstantFusion) {
        pm.addPass(VPUIP::createPatchFusedConstantsPass(log));
    }

    VPUIP::buildHardwareAdaptationPipeline(pm, log);

    // Handle WeightsTable, which requires statically allocated memory
    pm.addPass(VPUIP::createPatchWeightsTablePass(log));

    // Level 1 : VPU RunTime

    if (options.enableProfiling && options.enableSWProfiling) {
        pm.addPass(VPUIP::createUPAProfilingPass(log));
        pm.addPass(createMoveDeclarationsToTopPass(log));
    }

    pm.addPass(VPUIP::createConvertFuncArgsToDeclarationsPass(log));
    pm.addPass(VPUIP::arch30xx::createUnrollClusterTilingPass(log));
    pm.addPass(VPUIP::createNNDMATilingPass(log));

    pm.addPass(VPUIP::createDMABarrierOptimizationPass(log));

    VPURT::buildBarrierLegalizationPipeline(pm, log);

    if (options.enableWeightsSparsity) {
        pm.addPass(VPUIP::createFlattenSparseWeightsTypesPass(log));
    }
    if (options.enableSEPtrsOperations) {
        pm.addPass(VPUIP::createComputeSESizesPass(/*onlyInputsConcatOverC=*/false, log));
        pm.addPass(VPUIP::createAdjustInputDataForExplicitSETablePass(log));
    }

    pm.addPass(VPUIP::createUnwrapClusterTilingPass(log));

    pm.addPass(VPUIP::createUnrollDepthToSpaceDMAPass(log));
    pm.addPass(VPUIP::createUnrollSpaceToDepthDMAPass(log));
    pm.addPass(VPUIP::createUnrollPermuteToNNDMAPass(log));

    pm.addPass(VPUIP::createUnrollExpandDMAPass(log));
    pm.addPass(VPUIP::createDMABarrierOptimizationPass(log));

    if (options.enableProfiling) {
        if (options.enableDMAProfiling) {
            pm.addPass(VPUIP::createDMATaskProfilingAfterBarrierSchedPass(log));
        }
        pm.addPass(VPUIP::createGroupProfilingBuffersPass(log));
        pm.addPass(createMoveDeclarationsToTopPass(log));
    }

    pm.addPass(VPURT::createAssignPhysicalBarriersPass(log));
    pm.addPass(VPURT::createBarrierSimulationPass(log));
    pm.addPass(VPUIP::createDumpStatisticsOfTaskOpsPass(false, log));
    pm.addPass(Const::createConstantFoldingPass());
}

//
// ShaveCodeGen
//

void vpux::buildShaveCodeGenPipeline30XX(mlir::OpPassManager& pm, Logger log) {
    log.trace("Entered buildShaveCodeGenPipeline30XX()");

    // Code copied from the buildDefaultHWModePipeline().
    const auto grc = getDefaultGreedyRewriteConfig();

    // Level 3 : Topology

    pm.addPass(IE::createConvertScalarToTensorPass(log));
    pm.addPass(IE::createNormalizeL2FusionPass(log));
    pm.addPass(IE::createMatMulInputsTo2dPass(log));
    pm.addPass(IE::createConvertMatMulToConvPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    const DefaultHWOptions30XX options;  // TODO: takeout (normally)

    IE::buildAdjustPrecisionPipeline(pm, log);

    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));
    if (options.enableConvertFCToConv) {
        pm.addPass(IE::createConvertFCToConvPass(log));
    }

    IE::buildOperationConversionPipeline(pm, log);

    pm.addPass(IE::createConvertNceOpsTo4DPass(log));
    if (options.enableHandleLargeKernel) {
        pm.addPass(IE::createHandleLargeKernelsPass(log));
    }
    pm.addPass(IE::createHandleExcludePadForAvgPoolPass(log));
    if (options.enableConvertAvgPoolToDWConv) {
        pm.addPass(IE::createConvertAvgPoolToDWConvPass(log));
    }

    pm.addPass(IE::createAdaptShapesForScaleShiftPass(log));
    pm.addPass(IE::createResolveStridedSlicePass(log));
    pm.addPass(IE::createConvertShapeTo4DPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createSwapOperationsPass(log));
    pm.addPass(IE::createSwapPadLayerPass(log));
    pm.addPass(IE::createConvertSubtractToAddPass(log));
    pm.addPass(IE::createConvertToScaleShiftPass(log));
    pm.addPass(IE::createBroadcastInputForAddPass(log));
    pm.addPass(IE::createConvertGRNToNormalizeL2Pass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createResolveScatterUpdateByTransposePass(log));
    pm.addPass(IE::createConvertGroupConvToConvPass(log));
    pm.addPass(IE::createSwapOperationsPass(log));

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
    pm.addPass(mlir::createCanonicalizerPass(grc));
    if (options.enableOptimizeScaleShiftToDWConv) {
        IE::buildScaleShiftProcessingPipeline(pm, log);
    }

    pm.addPass(IE::createFusePostOpsPass(log));
    if (options.enableLowPrecision) {
        IE::buildLowPrecisionPipeline(pm, IE::LowPrecisionOptions(options), log);
        pm.addPass(IE::createConvertShapeTo4DPass(log));
        pm.addPass(IE::createSwapQuantCastAndClampPass(log));
    }
    pm.addPass(IE::createConvertStridedSlice2DWConvPass(log));
    pm.addPass(IE::createFusePostOpsPass(log));

    if (options.enableBilinearInterpolateOnDPU) {
        pm.addPass(IE::arch30xx::createMapBilinearInterpolateOnDPUPass(isOptionEnabled(options.enableSEPtrsOperations),
                                                                       log));
    }

    pm.addPass(IE::createConvertBatchedConvTo1NPass(log));
    pm.addPass(IE::arch30xx::createUnrollBatchPass(log));

    if (options.enableUpstreamSlice) {
        pm.addPass(IE::createUpstreamSlicePass(log));
    }

    pm.addPass(IE::createSwapMVNWithTransposePass(log));

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);

    if (options.enableExpandActivationChannels) {
        pm.addPass(IE::arch30xx::createExpandActivationChannelsPass(
                /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations), log));
        pm.addPass(mlir::createCanonicalizerPass(grc));

        if (options.enableOptimizeSliceExpand) {
            pm.addPass(IE::arch30xx::createOptimizeSliceExpandPass(log));
        }

        pm.addPass(IE::createAdjustInputShapeForEltwisePass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
        if (options.enableOptimizeSliceExpand) {
            pm.addPass(IE::arch30xx::createOptimizeSliceExpandPass(log));
        }

        if (options.enableOptimizeReorders) {
            pm.addPass(IE::createOptimizeReordersPass(/*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
                                                      log));
            pm.addPass(IE::createUniquifyOpsPass(log));
            pm.addPass(IE::createPropagateAffineReshapePass(log));
            pm.addPass(IE::createUniquifyBranchesPass(log));
        }
    }

    pm.addPass(IE::createSwapOperationsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createMovePermutePostEltwisePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    // Here it ends the code added from buildDefaultHWModePipeline().

    // Here we add new code
    buildLowerIE2IERTPipeline(pm, log);

    pm.addPass(vpux::createConvertSWLayers2AffinePass(log));
    pm.addPass(vpux::createConvertAffine2LLVMPass(log));

    // TODO: lowering to shave ASM
    // TODO: need pass IERT to VPUIP for e.g. function @main

    log.trace("Exiting buildShaveCodeGenPipeline30XX()");
}

//
// DefaultHWMode
//

void vpux::buildDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions30XX& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // Level 3 : Topology
    if (options.logOpOptimizations) {
        pm.addPass(IE::createLogOpOptimizationsPass());
    }
    pm.addPass(IE::createNormalizeL2FusionPass(log));
    pm.addPass(IE::createMatMulInputsTo2dPass(log));
    pm.addPass(IE::createConvertMatMulToConvPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustPrecisionPipeline(pm, log);

    if (options.enableConvertFCToConv) {
        pm.addPass(IE::createConvertFCToConvPass(log));
    }

    IE::buildOperationConversionPipeline(pm, log);

    pm.addPass(IE::createConvertNceOpsTo4DPass(log));
    if (options.enableHandleLargeKernel) {
        pm.addPass(IE::createHandleLargeKernelsPass(log));
    }
    pm.addPass(IE::createHandleExcludePadForAvgPoolPass(log));
    if (options.enableConvertAvgPoolToDWConv) {
        pm.addPass(IE::createConvertAvgPoolToDWConvPass(log));
    }

    pm.addPass(IE::createAdaptShapesForScaleShiftPass(log));
    pm.addPass(IE::createResolveStridedSlicePass(log));
    pm.addPass(IE::createConvertShapeTo4DPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createSwapOperationsPass(log));
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
    pm.addPass(IE::createSwapOperationsPass(log));

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
    pm.addPass(mlir::createCanonicalizerPass(grc));
    if (options.enableOptimizeScaleShiftToDWConv) {
        IE::buildScaleShiftProcessingPipeline(pm, log);
    }

    pm.addPass(IE::createFusePostOpsPass(log));
    if (options.enableLowPrecision) {
        IE::buildLowPrecisionPipeline(pm, IE::LowPrecisionOptions(options), log);
        pm.addPass(IE::createConvertShapeTo4DPass(log));
    }
    pm.addPass(IE::createConvertStridedSlice2DWConvPass(log));
    IE::arch30xx::buildOptimizeActivationsPipeline(pm, log);

    if (options.enableBilinearInterpolateOnDPU) {
        pm.addPass(IE::arch30xx::createMapBilinearInterpolateOnDPUPass(isOptionEnabled(options.enableSEPtrsOperations),
                                                                       log));
    }

    pm.addPass(IE::createConvertBatchedConvTo1NPass(log));
    pm.addPass(IE::arch30xx::createUnrollBatchPass(log));

    if (options.enableUpstreamSlice) {
        pm.addPass(IE::createUpstreamSlicePass(log));
    }

    pm.addPass(IE::createSwapMVNWithTransposePass(log));

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);
    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));

    if (options.enableExpandActivationChannels) {
        pm.addPass(IE::arch30xx::createExpandActivationChannelsPass(
                /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations), log));
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
            pm.addPass(IE::createOptimizeReordersPass(/*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
                                                      log));
            pm.addPass(IE::createUniquifyOpsPass(log));
            pm.addPass(IE::createPropagateAffineReshapePass(log));
            pm.addPass(IE::createUniquifyBranchesPass(log));
        }
    }

    pm.addPass(IE::createSwapOperationsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createMovePermutePostEltwisePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createPropagateMemPermuteThroughAddPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createUniquifyOpsPass(log));
    pm.addPass(IE::createRemoveIdentityPoolPass(log));
    pm.addPass(IE::createConvertExpandToConvPass(log));
    if (options.logOpOptimizations) {
        pm.addPass(IE::createLogOpOptimizationsPass());
    }

    // Lowering to VPU
    vpux::arch30xx::buildLowerIE2VPUPipeline30XX(pm, log);

    pm.addPass(VPU::createResolvePWLPostOpsPass(log));

    if (options.enableSEPtrsOperations) {
        pm.addPass(VPU::createLowerOpsToSENCEPass(log));
    }
    if (options.enableWeightsSparsity) {
        VPU::buildWeightsSparsityPipeline(pm, VPU::WeightsSparsityOptions(options), log);
    }

    pm.addPass(VPU::createEnsureNCEOpsSizeRequirementsPass(log));
    if (options.enableInPlaceEltwise) {
        pm.addPass(VPU::createDetectInPlaceEltwisePass(log));
    }

    if (options.enableSMPipeline) {
        VPU::buildSMPipeline(pm, VPU::TilingOptions(options), log);
    } else {
        VPU::arch30xx::buildIncrementalPipeline(pm, VPU::TilingOptions(options), log);
    }

    pm.addPass(VPU::createOptimizeConcatPass(log));
    pm.addPass(VPU::createAdjustMemorySpacePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPU::createCMXConcatPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPU::createSplitNCEOpsOntoWorkloadsPass(log));
    pm.addPass(VPU::createResolveEltwiseWithZTiledWorkloadsPass(log));

    // Lowering to VPUIP
    vpux::arch30xx::buildLowerVPU2VPUIP30XXPipeline(pm, log);
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(VPUIP::createConvertExpandPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPUIP::createConvertEltwiseToInPlacePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // Level 2 : Abstract RunTime

    pm.addPass(VPUIP::createSetMemorySpacePass(getMemKind<VPU::MemoryKind::DDR>, log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableOptimizeCopies) {
        pm.addPass(VPUIP::createMovePureViewOpBeforeCopyPass(log));
        pm.addPass(VPUIP::createOptimizeCopiesPass(log));
        pm.addPass(VPUIP::createOptimizeConcatViewCopiesPass(log));
        pm.addPass(VPUIP::createFuseDDRCopiesIntoConcats(log));
        pm.addPass(VPUIP::createCopyOpHoistingPass(log));
        pm.addPass(VPUIP::createOptimizeParallelCopiesPass(options.enableOptimizeConstCopies, log));
        pm.addPass(VPUIP::createMovePureViewOpBeforeCopyPass(log));
    }

    pm.addPass(VPUIP::createConvertToDMAPass(log));
    pm.addPass(VPUIP::createCopyOpTilingPass(log));

    if (options.enableSEPtrsOperations) {
        pm.addPass(VPUIP::createMoveSubViewBeforeSparseBufferPass(log));
        pm.addPass(VPUIP::createComputeSEBasePtrsPass(log));
        pm.addPass(VPUIP::createConvertSETablesToConstantsPass(log));
    }
    if (options.enableWeightsSparsity) {
        pm.addPass(VPUIP::createPropagateCompressionSchemePass(log));
    }
    if (options.enableWeightsSparsity) {
        pm.addPass(VPUIP::createUngroupSparseBuffersPass(log));
    }

    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableConstantFusion) {
        pm.addPass(VPUIP::createFuseConstantsPass(log));
    }

    if (options.enableProfiling && options.enableDPUProfiling) {
        pm.addPass(VPUIP::createDPUProfilingPass(getMemKind<VPU::MemoryKind::CMX_NN>, log));
    }
    if (options.enableProfiling && options.enableSWProfiling) {
        pm.addPass(VPUIP::createActShaveProfilingPass(getMemKind<VPU::MemoryKind::CMX_NN>, log));
    }

    VPUIP::buildAsyncSchedulingPipeline(pm, log);

    if (options.enableProfiling && options.enableDMAProfiling) {
        pm.addPass(VPUIP::createDMATaskProfilingReserveMemPass(log));
    }

    pm.addPass(VPUIP::createCalculateAsyncRegionCycleCostPass(log));

    pm.addPass(VPUIP::createFeasibleAllocationPass(getMemKind<VPU::MemoryKind::CMX_NN>,
                                                   getMemKind<VPU::MemoryKind::DDR>, log));

    pm.addPass(VPUIP::createMaximizeUPACyclesPass(log));

    if (options.enableGroupAsyncExecuteOps) {
        pm.addPass(VPUIP::createGroupAsyncExecuteOpsPass(log));
    }

    pm.addPass(VPUIP::createStaticAllocationPass(getMemKind<VPU::MemoryKind::DDR>, log));
    pm.addPass(VPUIP::createOptimizeAsyncDepsPass(log));
    pm.addPass(VPUIP::createBreakDataFlowPass(log));

    if (options.enableConstantFusion) {
        pm.addPass(VPUIP::createPatchFusedConstantsPass(log));
    }

    VPUIP::buildHardwareAdaptationPipeline(pm, log);

    // Handle WeightsTable, which requires statically allocated memory
    pm.addPass(VPUIP::createPatchWeightsTablePass(log));

    // Level 1 : VPU RunTime

    if (options.enableProfiling && options.enableSWProfiling) {
        pm.addPass(VPUIP::createUPAProfilingPass(log));
        pm.addPass(createMoveDeclarationsToTopPass(log));
    }

    pm.addPass(VPUIP::createConvertFuncArgsToDeclarationsPass(log));
    pm.addPass(VPUIP::createUnrollSwKernelPass(log));
    pm.addPass(VPUIP::arch30xx::createUnrollClusterTilingPass(log));
    pm.addPass(VPUIP::createNNDMATilingPass(log));

    pm.addPass(VPUIP::createDMABarrierOptimizationPass(log));

    VPURT::buildBarrierLegalizationPipeline(pm, log);

    if (options.enableWeightsSparsity) {
        pm.addPass(VPUIP::createFlattenSparseWeightsTypesPass(log));
    }
    if (options.enableSEPtrsOperations) {
        pm.addPass(VPUIP::createComputeSESizesPass(/*onlyInputsConcatOverC=*/false, log));
        pm.addPass(VPUIP::createAdjustInputDataForExplicitSETablePass(log));
    }

    pm.addPass(VPUIP::createUnwrapClusterTilingPass(log));

    pm.addPass(VPUIP::createUnrollDepthToSpaceDMAPass(log));
    pm.addPass(VPUIP::createUnrollSpaceToDepthDMAPass(log));
    pm.addPass(VPUIP::createUnrollPermuteToNNDMAPass(log));

    pm.addPass(VPUIP::createUnrollUpsamplingDMAPass(log));
    pm.addPass(VPUIP::createUnrollExpandDMAPass(log));
    pm.addPass(VPUIP::createUnrollPerAxisTileDMAPass(log));
    pm.addPass(VPUIP::createDMABarrierOptimizationPass(log));

    if (options.enableProfiling) {
        if (options.enableDMAProfiling) {
            pm.addPass(VPUIP::createDMATaskProfilingAfterBarrierSchedPass(log));
        }
        pm.addPass(VPUIP::createGroupProfilingBuffersPass(log));
        pm.addPass(createMoveDeclarationsToTopPass(log));
    }

    pm.addPass(VPURT::createAssignPhysicalBarriersPass(log));
    pm.addPass(VPURT::createBarrierSimulationPass(log));

    if (options.enableScheduleTrace) {
        pm.addPass(VPURT::createInferenceExecutionAnalysisPass(options.scheduleTraceFile, log));
    }

    pm.addPass(VPUIP::createDumpStatisticsOfTaskOpsPass(false, log));
    pm.addPass(Const::createConstantFoldingPass());
}
