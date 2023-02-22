//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/pipelines.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/dialect/EMU/passes.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPURT/passes.hpp"
#include "vpux/compiler/dialect/const/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/optional.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// Common utilities
//

namespace {

//
// Common utilities
//

template <VPU::MemoryKind KIND>
mlir::Optional<VPU::MemoryKind> getMemKind(StringRef) {
    return KIND;
}

VPU::ArchKind getArchKind(const StrOption& archKind) {
    VPUX_THROW_UNLESS(archKind.hasValue(), "Platform architecture is not provided. Please try 'vpu-arch=VPUX30XX'");
    const auto archKindStr = archKind.getValue();
    const auto parsed = VPU::symbolizeArchKind(archKindStr);
    VPUX_THROW_UNLESS(parsed.hasValue(), "Unsupported platform architecture '{0}'", archKindStr);
    return parsed.getValue();
}

Optional<int> getNumOfDPUGroups(const IntOption& numOfDPUGroups) {
    if (numOfDPUGroups.hasValue()) {
        return numOfDPUGroups.getValue();
    }
    return None;
}

Optional<int> getNumOfDMAPorts(const IntOption& numOfDMAPorts) {
    if (numOfDMAPorts.hasValue()) {
        return numOfDMAPorts.getValue();
    }
    return None;
}

}  // namespace

//
// ReferenceSWMode
//

void vpux::buildReferenceSWModePipeline(mlir::OpPassManager& pm, const ReferenceSWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // Level 3 : Topology

    pm.addPass(IE::createMatMulInputsTo2dPass(log));
    pm.addPass(IE::createConvertMatMulToConvPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    pm.addPass(IE::createConvertSquaredDiffToSubAndPowerPass(log));
    pm.addPass(IE::createResolveStridedSlicePass(log));
    pm.addPass(IE::createConvertShapeTo4DPass(log));
    pm.addPass(IE::createConvertBroadcastToTilePass(log));
    IE::buildAdjustForVPUPipeline(pm, IE::AdjustForVPUOptions(options), log);

    pm.addPass(IE::createSplitFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createDequantizeConstPass(log));
    pm.addPass(IE::createMergeFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);
    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));

    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(IE::createConvertReduceToPoolingPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // Lowering to VPU
    pm.addPass(createConvertLayers2VPUPass(log));

    pm.addPass(VPU::createConvertScalarToTensorPass(log));

    // Lowering to VPUIP
    pm.addPass(createBufferizeFuncAndReturnPass(log));
    pm.addPass(createAddBuffersForNetResults(log));

    pm.addPass(createConvertSWLayers2VPUIPPass(log));
    pm.addPass(createConvertLayers2VPUIPPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(createConvertSWLayers2VPUIPPass(log));

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

    if (options.enableCompressWeightsBTC) {
        pm.addPass(VPUIP::createCompressWeightsBTCPass(log));
    }

    pm.addPass(VPURT::createAssignPhysicalBarriersPass(log));
    pm.addPass(VPURT::createBarrierSimulationPass(log));
    pm.addPass(VPUIP::createDumpStatisticsOfTaskOpsPass(options.enableCompressWeightsBTC, log));
    pm.addPass(Const::createConstantFoldingPass());
}

//
// ReferenceHWMode
//

void vpux::buildReferenceHWModePipeline(mlir::OpPassManager& pm, const ReferenceHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // Level 3 : Topology

    pm.addPass(IE::createMatMulInputsTo2dPass(log));
    pm.addPass(IE::createConvertMatMulToConvPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    if (options.enableConvertFCToConv) {
        pm.addPass(IE::createConvertFCToConvPass(log));
    }
    pm.addPass(IE::createConvertReduceToPoolingPass(log));
    pm.addPass(IE::createConvertSquaredDiffToSubAndPowerPass(log));
    pm.addPass(IE::createConvertPowerToMultPass(log));

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
    pm.addPass(IE::createConvertBroadcastToTilePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createResolveScatterUpdateByTransposePass(log));

    IE::buildAdjustForVPUPipeline(pm, IE::AdjustForVPUOptions(options), log);

    if (options.enableSwapTransposeWithFQ) {
        pm.addPass(IE::createSwapTransposeWithFQPass(log));
    }
    if (options.enableConvertScaleShiftDW) {
        pm.addPass(IE::createConvertScaleShiftToDWPass(log));
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

    if (options.enableLowPrecision) {
        IE::buildLowPrecisionPipeline(pm, IE::LowPrecisionOptions(options), log);
        pm.addPass(IE::createConvertShapeTo4DPass(log));
        pm.addPass(IE::createSwapQuantCastAndClampPass(log));
        pm.addPass(IE::createInsertMaxpoolToConcatActivationPass(log));
    }
    pm.addPass(IE::createFusePostOpsPass(log));
    pm.addPass(IE::createConvertBatchedConvTo1NPass(log));
    pm.addPass(IE::createUnrollBatchPass(log));

    if (options.enableUpstreamSlice) {
        pm.addPass(IE::createUpstreamSlicePass(log));
    }

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);

    if (options.enableFusePermuteQuantize) {
        pm.addPass(IE::createFusePermuteQuantizePass(log));
        pm.addPass(IE::createConvertReorderToPermuteQuantizePass(log));
    }

    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));

    if (options.enableExpandActivationChannels) {
        pm.addPass(IE::createExpandActivationChannelsPass(log));
        pm.addPass(IE::createExpandActivationWidthPass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));

        if (options.enableOptimizeSliceExpand) {
            pm.addPass(IE::createOptimizeSliceExpandPass(log));
        }

        pm.addPass(IE::createAdjustInputShapeForEltwisePass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
        if (options.enableOptimizeSliceExpand) {
            pm.addPass(IE::createOptimizeSliceExpandPass(log));
        }

        if (options.enableOptimizeReorders) {
            pm.addPass(IE::createOptimizeReordersPass(log));
            pm.addPass(IE::createUniquifyOpsPass(log));
            pm.addPass(IE::createPropagateAffineReshapePass(log));
            pm.addPass(IE::createUniquifyBranchesPass(log));
        }

        if (options.enableFusePermuteQuantizeExpand) {
            pm.addPass(IE::createPropagateExpandPass(log));
            pm.addPass(IE::createFusePermuteQuantizeExpandPass(log));
        }
    }

    if (options.forceHostPrecisionLayoutConversion) {
        pm.addPass(IE::createForceHostPrecisionLayoutConversionPass(log));
    }

    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createLegalizeNDMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(IE::createMovePermutePostEltwisePass(log));

    // Lowering to VPU
    buildLowerIE2VPUPipeline(pm, log);

    pm.addPass(VPU::createResolvePWLPostOpsPass(log));
    pm.addPass(VPU::createSetupPPEPass(log));
    pm.addPass(VPU::createFuseClampPass(log));

    if (options.enableWeightsSparsity) {
        VPU::buildWeightsSparsityPipeline(pm, VPU::WeightsSparsityOptions(options), log);
    }
    if (options.enableActivationSparsity) {
        VPU::buildActivationSparsityPipeline(pm, VPU::ActivationSparsityOptions(options), log);
        pm.addPass(VPU::createLowerSparsityOpsPass(/*fakeSparsify=*/false, log));
    }

    if (options.enableInPlaceEltwise && !options.enableProfiling) {
        pm.addPass(VPU::createDetectInPlaceEltwisePass(log));
    }

    pm.addPass(VPU::createMultiClusterStrategyAssignmentPass(log));

    // manual strategy debug configuration
    StringRef writeStrategyFileLocation = "strategy_out.json";
    StringRef readStrategyFileLocation = "strategy_in.json";

    pm.addPass(VPU::createManualStrategyUtilsPass(options.writeStrategyToJson, writeStrategyFileLocation,
                                                  options.readStrategyFromJson, readStrategyFileLocation, log));

    pm.addPass(VPU::createManualTilingPass(log));

    if (options.enablePrefetchTiling) {
        pm.addPass(VPU::createPrefetchTilingPass(log));
    } else {
        pm.addPass(VPU::createIsolatedTilingPass(log));
    }

    pm.addPass(VPU::createEnsureNCEOpsSizeRequirementsPass(options.allowLargeInputChannels, log));
    pm.addPass(VPU::createOptimizeConcatSliceToSliceConcatPass(log));

    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPU::createManualStrategyUtilsPass(options.writeStrategyToJson, writeStrategyFileLocation,
                                                  options.readStrategyFromJson, readStrategyFileLocation, log));

    pm.addPass(VPU::createWrapVPUOpsInNCEClusterTilingPass(log));

    pm.addPass(VPU::createAdjustMemorySpacePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPU::createCMXConcatPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPU::createSplitNCEOpsOntoWorkloadsPass(log));
    pm.addPass(VPU::createCorrectNCEWorkloadsPass(log));

    pm.addPass(VPU::createConvertScalarToTensorPass(log));

    // Lowering to VPUIP
    buildLowerVPU2VPUIPPipeline(pm, log);
    pm.addPass(VPUIP::createWrapWithPermuteAsNNDMAPass(log));
    pm.addPass(VPUIP::createConvertExpandPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPUIP::createConvertEltwiseToInPlacePass(log));

    // Level 2 : Abstract RunTime

    pm.addPass(VPUIP::createSetMemorySpacePass(getMemKind<VPU::MemoryKind::DDR>, log));

    if (options.enableOptimizeCopies) {
        pm.addPass(VPUIP::createMovePureViewOpBeforeCopyPass(log));
        pm.addPass(VPUIP::createOptimizeCopiesPass(log));
        pm.addPass(VPUIP::createCopyOpHoistingPass(log));
        pm.addPass(VPUIP::createOptimizeParallelCopiesPass(log));
    }

    pm.addPass(VPUIP::createConvertToDMAPass(log));
    pm.addPass(VPUIP::createCopyOpTilingPass(log));

    if (options.enableWeightsSparsity) {
        pm.addPass(VPUIP::createPropagateCompressionSchemePass(log));
    }
    if (options.enableWeightsSparsity || options.enableActivationSparsity) {
        pm.addPass(VPUIP::createUngroupSparseBuffersPass(log));
    }

    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(VPUIP::createAdjustCompressConvInputsPass(log));

    if (options.enableActivationSparsity) {
        pm.addPass(VPUIP::createComputeSESizesPass(/*onlyInputsConcatOverC=*/true, log));
    }

    if (options.enableConstantFusion) {
        pm.addPass(VPUIP::createFuseConstantsPass(log));
    }
    pm.addPass(VPUIP::createSwizzlingPass(options.enableWeightsSwizzling, options.enableActivationSwizzling, log));

    if (options.enableProfiling && options.enableDPUProfiling) {
        pm.addPass(VPUIP::createDPUProfilingPass(getMemKind<VPU::MemoryKind::CMX_NN>, log));
    }
    if (options.enableProfiling && options.enableSWProfiling) {
        pm.addPass(VPUIP::createActShaveProfilingPass(getMemKind<VPU::MemoryKind::CMX_NN>, log));
    }

    VPUIP::buildAsyncSchedulingPipeline(pm, log);

    if (options.enableProfiling && options.enableDMAProfiling) {
        if (options.dmaProfilingAfterBarrierSched) {
            pm.addPass(VPUIP::createDMATaskProfilingReserveMemPass(log));
        } else {
            pm.addPass(VPUIP::createDMATaskProfilingPass(getMemKind<VPU::MemoryKind::CMX_NN>, log));
        }
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
    pm.addPass(VPURT::createAssignVirtualBarriersPass(log));
    pm.addPass(VPUIP::createUnrollClusterTilingPass(log));
    if (options.enableWeightsSparsity) {
        pm.addPass(VPUIP::createFlattenSparseWeightsTypesPass(log));
    }
    if (options.enableActivationSparsity) {
        pm.addPass(VPUIP::createComputeSESizesPass(/*onlyInputsConcatOverC=*/false, log));
    }

    pm.addPass(VPUIP::createUnrollDepthToSpaceDMAPass(log));
    pm.addPass(VPUIP::createUnrollSpaceToDepthDMAPass(log));
    pm.addPass(VPUIP::createUnrollPermuteToNNDMAPass(log));
    pm.addPass(VPUIP::createUnrollExpandDMAPass(log));
    pm.addPass(VPUIP::createDMABarrierOptimizationPass(log));
    if (options.enableProfiling) {
        if (options.enableDMAProfiling && options.dmaProfilingAfterBarrierSched) {
            pm.addPass(VPUIP::createDMATaskProfilingAfterBarrierSchedPass(log));
        }
        pm.addPass(VPUIP::createGroupProfilingBuffersPass(log));
        pm.addPass(createMoveDeclarationsToTopPass(log));
    }
    pm.addPass(VPUIP::createResolveDMAWithSwizzlingPass(log));

    if (options.enableCompressWeightsBTC) {
        pm.addPass(VPUIP::createCompressWeightsBTCPass(log));
    }

    pm.addPass(VPURT::createAssignPhysicalBarriersPass(log));
    pm.addPass(VPURT::createBarrierSimulationPass(log));
    pm.addPass(VPUIP::createDumpStatisticsOfTaskOpsPass(options.enableCompressWeightsBTC, log));
    pm.addPass(Const::createConstantFoldingPass());
}

//
// DefaultHWMode
//

void vpux::buildDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // Level 3 : Topology

    pm.addPass(IE::createMatMulInputsTo2dPass(log));
    pm.addPass(IE::createConvertMatMulToConvPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    if (options.enableConvertFCToConv) {
        pm.addPass(IE::createConvertFCToConvPass(log));
    }
    pm.addPass(IE::createConvertExtractImagePatchesPass(log));
    pm.addPass(IE::createConvertReduceToPoolingPass(log));
    pm.addPass(IE::createConvertSquaredDiffToSubAndPowerPass(log));
    pm.addPass(IE::createConvertPowerToMultPass(log));

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
    pm.addPass(IE::createConvertBroadcastToTilePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createResolveScatterUpdateByTransposePass(log));

    IE::buildAdjustForVPUPipeline(pm, IE::AdjustForVPUOptions(options), log);

    if (options.enableSwapTransposeWithFQ) {
        pm.addPass(IE::createSwapTransposeWithFQPass(log));
    }
    if (options.enableConvertScaleShiftDW) {
        pm.addPass(IE::createConvertScaleShiftToDWPass(log));
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

    if (options.enableLowPrecision) {
        IE::buildLowPrecisionPipeline(pm, IE::LowPrecisionOptions(options), log);
        pm.addPass(IE::createConvertShapeTo4DPass(log));
        pm.addPass(IE::createSwapQuantCastAndClampPass(log));
        pm.addPass(IE::createInsertMaxpoolToConcatActivationPass(log));
    }
    pm.addPass(IE::createFusePostOpsPass(log));
    pm.addPass(IE::createConvertBatchedConvTo1NPass(log));
    pm.addPass(IE::createUnrollBatchPass(log));

    if (options.enableUpstreamSlice) {
        pm.addPass(IE::createUpstreamSlicePass(log));
    }

    pm.addPass(IE::createSwapMVNWithTransposePass(log));

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);
    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));

    if (options.enableFusePermuteQuantize) {
        pm.addPass(IE::createFusePermuteQuantizePass(log));
        pm.addPass(IE::createConvertReorderToPermuteQuantizePass(log));
    }

    if (options.enableExpandActivationChannels) {
        pm.addPass(IE::createExpandActivationChannelsPass(log));
        pm.addPass(IE::createExpandActivationWidthPass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));

        if (options.enableOptimizeSliceExpand) {
            pm.addPass(IE::createOptimizeSliceExpandPass(log));
        }

        pm.addPass(IE::createAdjustInputShapeForEltwisePass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
        if (options.enableOptimizeSliceExpand) {
            pm.addPass(IE::createOptimizeSliceExpandPass(log));
        }

        if (options.enableOptimizeReorders) {
            pm.addPass(IE::createOptimizeReordersPass(log));
            pm.addPass(IE::createUniquifyOpsPass(log));
            pm.addPass(IE::createPropagateAffineReshapePass(log));
            pm.addPass(IE::createUniquifyBranchesPass(log));
        }

        if (options.enableFusePermuteQuantizeExpand) {
            pm.addPass(IE::createPropagateExpandPass(log));
            pm.addPass(IE::createFusePermuteQuantizeExpandPass(log));
        }
    }

    if (options.forceHostPrecisionLayoutConversion) {
        pm.addPass(IE::createForceHostPrecisionLayoutConversionPass(log));
    }

    pm.addPass(IE::createSwapOperationsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createMovePermutePostEltwisePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createLegalizeNDMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // Lowering to VPU
    buildLowerIE2VPUPipeline(pm, log);

    pm.addPass(VPU::createResolvePWLPostOpsPass(log));
    pm.addPass(VPU::createSetupPPEPass(log));
    pm.addPass(VPU::createFuseClampPass(log));

    if (options.enableWeightsSparsity) {
        VPU::buildWeightsSparsityPipeline(pm, VPU::WeightsSparsityOptions(options), log);
    }
    if (options.enableActivationSparsity) {
        VPU::buildActivationSparsityPipeline(pm, VPU::ActivationSparsityOptions(options), log);
        pm.addPass(VPU::createLowerSparsityOpsPass(/*fakeSparsify=*/false, log));
    }

    if (options.enableInPlaceEltwise && !options.enableProfiling) {
        pm.addPass(VPU::createDetectInPlaceEltwisePass(log));
    }

    pm.addPass(VPU::createMultiClusterStrategyAssignmentPass(log));

    // manual strategy debug configuration
    StringRef writeStrategyFileLocation = "strategy_out.json";
    StringRef readStrategyFileLocation = "strategy_in.json";

    pm.addPass(VPU::createManualStrategyUtilsPass(options.writeStrategyToJson, writeStrategyFileLocation,
                                                  options.readStrategyFromJson, readStrategyFileLocation, log));

    pm.addPass(VPU::createManualTilingPass(log));

    if (options.enablePrefetchTiling) {
        pm.addPass(VPU::createPrefetchTilingPass(log));
    } else {
        pm.addPass(VPU::createIsolatedTilingPass(log));
    }

    pm.addPass(VPU::createEnsureNCEOpsSizeRequirementsPass(options.allowLargeInputChannels, log));
    pm.addPass(VPU::createOptimizeConcatSliceToSliceConcatPass(log));

    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPU::createManualStrategyUtilsPass(options.writeStrategyToJson, writeStrategyFileLocation,
                                                  options.readStrategyFromJson, readStrategyFileLocation, log));

    pm.addPass(VPU::createWrapVPUOpsInNCEClusterTilingPass(log));

    pm.addPass(VPU::createAdjustMemorySpacePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPU::createCMXConcatPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPU::createSplitNCEOpsOntoWorkloadsPass(log));
    pm.addPass(VPU::createCorrectNCEWorkloadsPass(log));

    pm.addPass(VPU::createConvertScalarToTensorPass(log));

    // Lowering to VPUIP
    buildLowerVPU2VPUIPPipeline(pm, log);
    pm.addPass(VPUIP::createTileActShaveKernelTaskPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(VPUIP::createWrapWithPermuteAsNNDMAPass(log));
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
        pm.addPass(VPUIP::createOptimizeSpillingCopiesPass(log));
        pm.addPass(VPUIP::createCopyOpHoistingPass(log));
        pm.addPass(VPUIP::createOptimizeParallelCopiesPass(log));
    }

    pm.addPass(VPUIP::createConvertToDMAPass(log));
    pm.addPass(VPUIP::createCopyOpTilingPass(log));

    if (options.enableWeightsSparsity) {
        pm.addPass(VPUIP::createPropagateCompressionSchemePass(log));
    }
    if (options.enableWeightsSparsity || options.enableActivationSparsity) {
        pm.addPass(VPUIP::createUngroupSparseBuffersPass(log));
    }

    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(VPUIP::createAdjustCompressConvInputsPass(log));

    if (options.enableActivationSparsity) {
        pm.addPass(VPUIP::createComputeSESizesPass(/*onlyInputsConcatOverC=*/true, log));
    }

    if (options.enableConstantFusion) {
        pm.addPass(VPUIP::createFuseConstantsPass(log));
    }
    pm.addPass(VPUIP::createSwizzlingPass(options.enableWeightsSwizzling, options.enableActivationSwizzling, log));

    if (options.enableProfiling && options.enableDPUProfiling) {
        pm.addPass(VPUIP::createDPUProfilingPass(getMemKind<VPU::MemoryKind::CMX_NN>, log));
    }
    if (options.enableProfiling && options.enableSWProfiling) {
        pm.addPass(VPUIP::createActShaveProfilingPass(getMemKind<VPU::MemoryKind::CMX_NN>, log));
    }

    VPUIP::buildAsyncSchedulingPipeline(pm, log);

    if (options.enableProfiling && options.enableDMAProfiling) {
        if (options.dmaProfilingAfterBarrierSched) {
            pm.addPass(VPUIP::createDMATaskProfilingReserveMemPass(log));
        } else {
            pm.addPass(VPUIP::createDMATaskProfilingPass(getMemKind<VPU::MemoryKind::CMX_NN>, log));
        }
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
    pm.addPass(VPURT::createAssignVirtualBarriersPass(log));
    pm.addPass(VPUIP::createUnrollSwKernelPass(log));
    pm.addPass(VPUIP::createUnrollClusterTilingPass(log));
    if (options.enableWeightsSparsity) {
        pm.addPass(VPUIP::createFlattenSparseWeightsTypesPass(log));
    }
    if (options.enableActivationSparsity) {
        pm.addPass(VPUIP::createComputeSESizesPass(/*onlyInputsConcatOverC=*/false, log));
    }

    pm.addPass(VPUIP::createUnrollDepthToSpaceDMAPass(log));
    pm.addPass(VPUIP::createUnrollSpaceToDepthDMAPass(log));
    pm.addPass(VPUIP::createUnrollPermuteToNNDMAPass(log));
    pm.addPass(VPUIP::createUnrollUpsamplingDMAPass(log));
    pm.addPass(VPUIP::createUnrollExpandDMAPass(log));
    pm.addPass(VPUIP::createUnrollPerAxisTileDMAPass(log));
    pm.addPass(VPUIP::createDMABarrierOptimizationPass(log));
    if (options.enableProfiling) {
        if (options.enableDMAProfiling && options.dmaProfilingAfterBarrierSched) {
            pm.addPass(VPUIP::createDMATaskProfilingAfterBarrierSchedPass(log));
        }
        pm.addPass(VPUIP::createGroupProfilingBuffersPass(log));
        pm.addPass(createMoveDeclarationsToTopPass(log));
    }
    pm.addPass(VPUIP::createResolveDMAWithSwizzlingPass(log));

    if (options.enableCompressWeightsBTC) {
        pm.addPass(VPUIP::createCompressWeightsBTCPass(log));
    }

    pm.addPass(VPURT::createAssignPhysicalBarriersPass(log));
    pm.addPass(VPURT::createBarrierSimulationPass(log));
    pm.addPass(VPUIP::createDumpStatisticsOfTaskOpsPass(options.enableCompressWeightsBTC, log));
    pm.addPass(Const::createConstantFoldingPass());
}

//
// EMU ReferenceSWMode
//

void vpux::buildEMUReferenceSWModePipeline(mlir::OpPassManager& pm, const ReferenceSWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // Level 3 : Topology

    pm.addPass(IE::createMatMulInputsTo2dPass(log));
    pm.addPass(IE::createConvertMatMulToConvPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    pm.addPass(IE::createResolveStridedSlicePass(log));
    pm.addPass(IE::createConvertShapeTo4DPass(log));

    IE::buildAdjustForVPUPipeline(pm, IE::AdjustForVPUOptions(options), log);

    pm.addPass(IE::createSplitFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createDequantizeConstPass(log));
    pm.addPass(IE::createMergeFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);
    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));

    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(IE::createConvertReduceToPoolingPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(createConvertLayers2VPUPass(log));
    EMU::buildAdjustForEMU(pm, log);
}

//
// EMU ReferenceHWMode
//

void vpux::buildEMUReferenceHWModePipeline(mlir::OpPassManager& pm, const ReferenceHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // Level 3 : Topology

    pm.addPass(IE::createMatMulInputsTo2dPass(log));
    pm.addPass(IE::createConvertMatMulToConvPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    if (options.enableConvertFCToConv) {
        pm.addPass(IE::createConvertFCToConvPass(log));
    }
    pm.addPass(IE::createConvertExtractImagePatchesPass(log));
    pm.addPass(IE::createConvertReduceToPoolingPass(log));
    pm.addPass(IE::createHandleLargeKernelsPass(log));
    pm.addPass(IE::createHandleExcludePadForAvgPoolPass(log));
    if (options.enableConvertAvgPoolToDWConv) {
        pm.addPass(IE::createConvertAvgPoolToDWConvPass(log));
    }

    pm.addPass(IE::createResolveStridedSlicePass(log));
    pm.addPass(IE::createConvertShapeTo4DPass(log));
    pm.addPass(IE::createConvertSubtractToAddPass(log));
    pm.addPass(IE::createConvertToScaleShiftPass(log));
    pm.addPass(IE::createConvertBroadcastToTilePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustForVPUPipeline(pm, IE::AdjustForVPUOptions(options), log);

    if (options.enableSwapTransposeWithFQ) {
        pm.addPass(IE::createSwapTransposeWithFQPass(log));
    }
    if (options.enableConvertScaleShiftDW) {
        pm.addPass(IE::createConvertScaleShiftToDWPass(log));
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

    if (options.enableLowPrecision) {
        IE::buildLowPrecisionPipeline(pm, IE::LowPrecisionOptions(options), log);
        pm.addPass(IE::createConvertShapeTo4DPass(log));
        pm.addPass(IE::createSwapQuantCastAndClampPass(log));
        pm.addPass(IE::createInsertMaxpoolToConcatActivationPass(log));
    }
    pm.addPass(IE::createFusePostOpsPass(log));
    pm.addPass(IE::createConvertBatchedConvTo1NPass(log));
    pm.addPass(IE::createUnrollBatchPass(log));

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);
    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));

    if (options.enableExpandActivationChannels) {
        pm.addPass(IE::createExpandActivationChannelsPass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));

        if (options.enableOptimizeSliceExpand) {
            pm.addPass(IE::createOptimizeSliceExpandPass(log));
        }

        if (options.enableOptimizeReorders) {
            pm.addPass(IE::createOptimizeReordersPass(log));
            pm.addPass(IE::createUniquifyOpsPass(log));
            pm.addPass(IE::createUniquifyBranchesPass(log));
        }
    }

    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(createConvertIEToVPUNCEPass(log));
    pm.addPass(createConvertLayers2VPUPass(log));
    pm.addPass(VPU::createResolvePWLPostOpsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // EMU Dialect lowering
    EMU::buildAdjustForEMU(pm, log);
}

//
// EMUDefaultHWMode
//

void vpux::buildEMUDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // Level 3 : Topology

    pm.addPass(IE::createMatMulInputsTo2dPass(log));
    pm.addPass(IE::createConvertMatMulToConvPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    if (options.enableConvertFCToConv) {
        pm.addPass(IE::createConvertFCToConvPass(log));
    }
    pm.addPass(IE::createConvertExtractImagePatchesPass(log));
    pm.addPass(IE::createConvertReduceToPoolingPass(log));
    pm.addPass(IE::createConvertPowerToMultPass(log));

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
    pm.addPass(IE::createConvertBroadcastToTilePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustForVPUPipeline(pm, IE::AdjustForVPUOptions(options), log);

    if (options.enableSwapTransposeWithFQ) {
        pm.addPass(IE::createSwapTransposeWithFQPass(log));
    }
    if (options.enableConvertScaleShiftDW) {
        pm.addPass(IE::createConvertScaleShiftToDWPass(log));
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

    if (options.enableLowPrecision) {
        IE::buildLowPrecisionPipeline(pm, IE::LowPrecisionOptions(options), log);
        pm.addPass(IE::createConvertShapeTo4DPass(log));
        pm.addPass(IE::createSwapQuantCastAndClampPass(log));
        pm.addPass(IE::createInsertMaxpoolToConcatActivationPass(log));
    }
    pm.addPass(IE::createFusePostOpsPass(log));
    pm.addPass(IE::createConvertBatchedConvTo1NPass(log));
    pm.addPass(IE::createUnrollBatchPass(log));

    if (options.enableUpstreamSlice) {
        pm.addPass(IE::createUpstreamSlicePass(log));
    }

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);
    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));
    if (options.enableFusePermuteQuantize) {
        pm.addPass(IE::createFusePermuteQuantizePass(log));
        pm.addPass(IE::createConvertReorderToPermuteQuantizePass(log));
    }
    if (options.enableExpandActivationChannels) {
        pm.addPass(IE::createExpandActivationChannelsPass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));

        if (options.enableOptimizeSliceExpand) {
            pm.addPass(IE::createOptimizeSliceExpandPass(log));
        }

        if (options.enableOptimizeReorders) {
            pm.addPass(IE::createOptimizeReordersPass(log));
            pm.addPass(IE::createUniquifyOpsPass(log));
            pm.addPass(IE::createUniquifyBranchesPass(log));
        }

        if (options.enableFusePermuteQuantizeExpand) {
            pm.addPass(IE::createPropagateExpandPass(log));
            pm.addPass(IE::createFusePermuteQuantizeExpandPass(log));
        }
    }

    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(createConvertIEToVPUNCEPass(log));
    pm.addPass(createConvertLayers2VPUPass(log));
    pm.addPass(VPU::createResolvePWLPostOpsPass(log));

    // EMU Dialect lowering
    EMU::buildAdjustForEMU(pm, log);
}
//
// registerPipelines
//

void vpux::registerPipelines() {
    mlir::PassPipelineRegistration<ReferenceSWOptions>(
            "reference-sw-mode", "Compile IE Network in Reference Software mode (SW only execution)",
            [](mlir::OpPassManager& pm, const ReferenceSWOptions& options) {
                const auto archKind = getArchKind(options.arch);
                pm.addPass(VPU::createInitCompilerPass(archKind, VPU::CompilationMode::ReferenceSW));

                buildReferenceSWModePipeline(pm, options);
            });

    mlir::PassPipelineRegistration<ReferenceHWOptions>(
            "reference-hw-mode", "Compile IE Network in Reference Hardware mode (HW and SW execution)",
            [](mlir::OpPassManager& pm, const ReferenceHWOptions& options) {
                const auto archKind = getArchKind(options.arch);
                const auto numOfDPUGroups = getNumOfDPUGroups(options.numberOfDPUGroups);
                const auto numOfDMAPorts = getNumOfDMAPorts(options.numberOfDMAPorts);
                pm.addPass(VPU::createInitCompilerPass(archKind, VPU::CompilationMode::ReferenceHW, numOfDPUGroups,
                                                       numOfDMAPorts));

                buildReferenceHWModePipeline(pm, options);
            });

    mlir::PassPipelineRegistration<DefaultHWOptions>(
            "default-hw-mode", "Compile IE Network in Default Hardware mode (HW and SW execution)",
            [](mlir::OpPassManager& pm, const DefaultHWOptions& options) {
                const auto archKind = getArchKind(options.arch);
                const auto numOfDPUGroups = getNumOfDPUGroups(options.numberOfDPUGroups);
                const auto numOfDMAPorts = getNumOfDMAPorts(options.numberOfDMAPorts);
                pm.addPass(VPU::createInitCompilerPass(archKind, VPU::CompilationMode::DefaultHW, numOfDPUGroups,
                                                       numOfDMAPorts));

                buildDefaultHWModePipeline(pm, options);
            });

    mlir::PassPipelineRegistration<ReferenceSWOptions>(
            "emu-reference-sw-mode", "Compile IE Network in EMU Reference Software mode (SW only execution)",
            [](mlir::OpPassManager& pm, const ReferenceSWOptions& options) {
                const auto archKind = getArchKind(options.arch);
                pm.addPass(VPU::createInitCompilerPass(archKind, VPU::CompilationMode::ReferenceSW));

                buildEMUReferenceSWModePipeline(pm, options);
            });
}
