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
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    IE::buildAdjustForVPUPipeline(pm, IE::AdjustForVPUOptions(options), log);

    pm.addPass(IE::createSplitFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createDequantizeConstPass(log));
    pm.addPass(IE::createMergeFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);

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

    pm.addPass(VPUIP::createSetInternalMemorySpacePass(getMemKind<VPU::MemoryKind::DDR>, log));

    pm.addPass(VPUIP::createCopyOpTilingPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    VPUIP::buildAsyncSchedulingPipeline(pm, log);

    pm.addPass(VPUIP::createStaticAllocationPass(getMemKind<VPU::MemoryKind::CMX_NN>, log));
    pm.addPass(VPUIP::createStaticAllocationPass(getMemKind<VPU::MemoryKind::DDR>, log));
    pm.addPass(VPUIP::createLinearizationPass(log));
    pm.addPass(VPUIP::createOptimizeAsyncDepsPass(log));

    pm.addPass(VPUIP::createBreakDataFlowPass(log));

    VPUIP::buildHardwareAdaptationPipeline(pm, VPUIP::HardwareAdaptationOptions(options), log);

    // Level 1 : VPU RunTime

    if (options.enableProfiling) {
        if (options.enableSWProfiling) {
            pm.addPass(VPUIP::createUPAProfilingPass(log));
        }
        pm.addPass(VPUIP::createGroupProfilingBuffersPass(log));
        pm.addPass(createMoveDeclarationsToTopPass(log));
    }

    pm.addPass(VPURT::createAssignPhysicalBarriersPass(log));
    pm.addPass(VPURT::createBarrierSimulationPass(log));
    pm.addPass(VPUIP::createDumpStatisticsOfTaskOpsPass(log));
    pm.addPass(Const::createConstantFoldingPass());
}

//
// ReferenceHWMode
//

void vpux::buildReferenceHWModePipeline(mlir::OpPassManager& pm, const ReferenceHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // Level 3 : Topology

    pm.addPass(IE::createMatMulInputsTo2dPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    pm.addPass(IE::createUnrollBatchPass(log));

    if (options.enableConvertFCToConv) {
        pm.addPass(IE::createConvertFCToConvPass(log));
    }
    pm.addPass(IE::createConvertReduceToPoolingPass(log));
    pm.addPass(IE::createHandleLargeKernelsPass(log));
    pm.addPass(IE::createHandleExcludePadForAvgPoolPass(log));
    if (options.enableConvertAvgPoolToDWConv) {
        pm.addPass(IE::createConvertAvgPoolToDWConvPass(log));
    }

    pm.addPass(IE::createConvertSubtractToNegativeAddPass(log));
    pm.addPass(IE::createConvertToScaleShiftPass(log));
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
    }
    pm.addPass(IE::createFusePostOpsPass(log));

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);

    if (options.enableExpandActivationChannels) {
        pm.addPass(IE::createExpandActivationChannelsPass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));

        if (options.enableOptimizeReorders) {
            pm.addPass(IE::createOptimizeReordersPass(log));
            pm.addPass(IE::createUniquifyOpsPass(log));
        }
    }

    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // Lowering to VPU
    pm.addPass(createConvertIEToVPUNCEPass(log));
    pm.addPass(createConvertIEToVPUM2IPass(log));
    pm.addPass(createConvertLayers2VPUPass(log));

    pm.addPass(VPU::createResolvePWLPostOpsPass(log));
    pm.addPass(VPU::createSetupPPEPass(log));
    pm.addPass(VPU::createFuseM2IOpsPass(log));
    pm.addPass(VPU::createConvertM2IOpsPass(log));

    pm.addPass(VPU::createIsolatedTilingPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPU::createAdjustMemorySpacePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPU::createSplitNCEOpsOntoWorkloadsPass(log));

    pm.addPass(VPU::createConvertScalarToTensorPass(log));

    // Lowering to VPUIP
    buildLowerIERT2VPUIPPipeline(pm, log);

    // Level 2 : Abstract RunTime

    pm.addPass(VPUIP::createSetInternalMemorySpacePass(getMemKind<VPU::MemoryKind::DDR>, log));

    pm.addPass(VPUIP::createCopyOpTilingPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableProfiling && options.enableDPUProfiling) {
        pm.addPass(VPUIP::createDPUProfilingPass(getMemKind<VPU::MemoryKind::CMX_NN>, log));
    }

    VPUIP::buildAsyncSchedulingPipeline(pm, log);

    if (options.enableProfiling && options.enableDMAProfiling) {
        pm.addPass(VPUIP::createDMATaskProfilingPass(getMemKind<VPU::MemoryKind::CMX_NN>, log));
    }

    pm.addPass(VPUIP::createStaticAllocationPass(getMemKind<VPU::MemoryKind::CMX_NN>, log));
    pm.addPass(VPUIP::createStaticAllocationPass(getMemKind<VPU::MemoryKind::DDR>, log));
    pm.addPass(VPUIP::createLinearizationPass(log));
    pm.addPass(VPUIP::createOptimizeAsyncDepsPass(log));

    pm.addPass(VPUIP::createBreakDataFlowPass(log));

    VPUIP::buildHardwareAdaptationPipeline(pm, VPUIP::HardwareAdaptationOptions(options), log);

    pm.addPass(VPUIP::createPatchWeightsTablePass(log));

    // Level 1 : VPU RunTime

    if (options.enableProfiling) {
        if (options.enableSWProfiling) {
            pm.addPass(VPUIP::createUPAProfilingPass(log));
        }
        pm.addPass(VPUIP::createGroupProfilingBuffersPass(log));
    }
    pm.addPass(VPURT::createAssignPhysicalBarriersPass(log));
    pm.addPass(VPURT::createBarrierSimulationPass(log));
    pm.addPass(VPUIP::createDumpStatisticsOfTaskOpsPass(log));
    pm.addPass(Const::createConstantFoldingPass());
}

//
// DefaultHWMode
//

void vpux::buildDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // Level 3 : Topology

    pm.addPass(IE::createMatMulInputsTo2dPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    pm.addPass(IE::createUnrollBatchPass(log));
    if (options.enableConvertFCToConv) {
        pm.addPass(IE::createConvertFCToConvPass(log));
    }
    pm.addPass(IE::createConvertReduceToPoolingPass(log));
    pm.addPass(IE::createConvertPowerToMultPass(log));

    if (options.enableHandleLargeKernel) {
        pm.addPass(IE::createHandleLargeKernelsPass(log));
    }
    pm.addPass(IE::createHandleExcludePadForAvgPoolPass(log));
    if (options.enableConvertAvgPoolToDWConv) {
        pm.addPass(IE::createConvertAvgPoolToDWConvPass(log));
    }

    pm.addPass(IE::createConvertSubtractToNegativeAddPass(log));
    pm.addPass(IE::createConvertToScaleShiftPass(log));
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
    }
    pm.addPass(IE::createFusePostOpsPass(log));
    pm.addPass(IE::createUnrollBatchPass(log));

    if (options.enableUpstreamSlice) {
        pm.addPass(IE::createUpstreamSlicePass(log));
    }

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);

    if (options.enableExpandActivationChannels) {
        pm.addPass(IE::createExpandActivationChannelsPass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));

        if (options.enableOptimizeReorders) {
            pm.addPass(IE::createOptimizeReordersPass(log));
            pm.addPass(IE::createUniquifyOpsPass(log));
        }
    }

    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // Lowering to VPU
    pm.addPass(createConvertIEToVPUNCEPass(log));
    pm.addPass(createConvertIEToVPUM2IPass(log));
    pm.addPass(createConvertLayers2VPUPass(log));

    pm.addPass(VPU::createResolvePWLPostOpsPass(log));
    pm.addPass(VPU::createSetupPPEPass(log));
    pm.addPass(VPU::createMultiClusterStrategyAssignmentPass(log));
    pm.addPass(VPU::createFuseM2IOpsPass(log));
    pm.addPass(VPU::createConvertM2IOpsPass(log));

    // manual strategy debug configuration
    bool writeStrategyToJSON = false;
    StringRef writeStrategyFileLocation = "strategy_out.json";
    bool readStrategyFromJSON = false;
    StringRef readStrategyFileLocation = "strategy_in.json";

    pm.addPass(VPU::createManualStrategyUtilsPass(writeStrategyToJSON, writeStrategyFileLocation, readStrategyFromJSON,
                                                  readStrategyFileLocation, log));

    pm.addPass(VPU::createManualTilingPass(log));

    // Disable prefetch tiling for VPUX37XX. This logic must be updated because for VPUX37XX, it is creating tiles even if is not
    // needed. [Track number: E#46807]
    if (options.enablePrefetchTiling) {
        pm.addPass(VPU::createPrefetchTilingPass(log));
    } else {
        pm.addPass(VPU::createIsolatedTilingPass(log));
    }

    pm.addPass(VPU::createOptimizeConcatSliceToSliceConcatPass(log));

    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPU::createManualStrategyUtilsPass(writeStrategyToJSON, writeStrategyFileLocation, readStrategyFromJSON,
                                                  readStrategyFileLocation, log));

    pm.addPass(VPU::createWrapVPUOpsInNCEClusterTilingPass(log));

    pm.addPass(VPU::createAdjustMemorySpacePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPU::createCMXConcatPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPU::createSplitNCEOpsOntoWorkloadsPass(log));
    pm.addPass(VPU::createCorrectNCEWorkloadsPass(log));

    pm.addPass(VPU::createConvertScalarToTensorPass(log));

    // Lowering to VPUIP
    buildLowerIERT2VPUIPPipeline(pm, log);

    // Level 2 : Abstract RunTime

    pm.addPass(VPUIP::createSetInternalMemorySpacePass(getMemKind<VPU::MemoryKind::DDR>, log));

    if (options.enableOptimizeCopies) {
        pm.addPass(VPUIP::createOptimizeCopiesPass(log));
        pm.addPass(VPUIP::createCopyOpHoistingPass(log));
        pm.addPass(VPUIP::createOptimizeParallelCopiesPass(log));
    }

    pm.addPass(VPUIP::createMoveOperationFromDDRtoCMXPass(log));

    pm.addPass(VPUIP::createCopyOpTilingPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(
            VPUIP::createAlignmentForSwizzling(options.enableWeightsSwizzling, options.enableActivationSwizzling, log));

    if (options.enableProfiling && options.enableDPUProfiling) {
        pm.addPass(VPUIP::createDPUProfilingPass(getMemKind<VPU::MemoryKind::CMX_NN>, log));
    }
    if (options.enableProfiling && options.enableSWProfiling) {
        pm.addPass(VPUIP::createActShaveProfilingPass(getMemKind<VPU::MemoryKind::CMX_NN>, log));
    }

    VPUIP::buildAsyncSchedulingPipeline(pm, log);

    if (options.enableProfiling && options.enableDMAProfiling) {
        pm.addPass(VPUIP::createDMATaskProfilingPass(getMemKind<VPU::MemoryKind::CMX_NN>, log));
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

    VPUIP::buildHardwareAdaptationPipeline(pm, VPUIP::HardwareAdaptationOptions(options), log);

    // Handle WeightsTable, which requires statically allocated memory
    pm.addPass(VPUIP::createPatchWeightsTablePass(log));
    if (options.enableWeightsSwizzling) {
        pm.addPass(VPUIP::createSwizzleConstantPass(log));
    }
    // Level 1 : VPU RunTime

    if (options.enableProfiling) {
        if (options.enableSWProfiling) {
            pm.addPass(VPUIP::createUPAProfilingPass(log));
        }
        pm.addPass(VPUIP::createGroupProfilingBuffersPass(log));
        pm.addPass(createMoveDeclarationsToTopPass(log));
    }

    pm.addPass(VPURT::createAssignVirtualBarriersPass(log));
    pm.addPass(VPUIP::createUnrollClusterTilingPass(log));
    pm.addPass(VPUIP::createUnrollDepthToSpaceDMAPass(log));
    pm.addPass(VPUIP::createUnrollPermuteToNNDMAPass(log));
    pm.addPass(VPUIP::createDMABarrierOptimizationPass(log));
    pm.addPass(VPURT::createAssignPhysicalBarriersPass(log));
    pm.addPass(VPURT::createBarrierSimulationPass(log));
    pm.addPass(VPUIP::createDumpStatisticsOfTaskOpsPass(log));
    pm.addPass(Const::createConstantFoldingPass());
}

//
// EMU ReferenceSWMode
//

void vpux::buildEMUReferenceSWModePipeline(mlir::OpPassManager& pm, const ReferenceSWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // Level 3 : Topology

    pm.addPass(IE::createMatMulInputsTo2dPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    IE::buildAdjustForVPUPipeline(pm, IE::AdjustForVPUOptions(options), log);

    pm.addPass(IE::createSplitFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createDequantizeConstPass(log));
    pm.addPass(IE::createMergeFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);

    EMU::buildAdjustForEMU(pm, log);
    pm.addPass(createConvertLayers2VPUPass(log));
}

//
// EMU ReferenceHWMode
//

void vpux::buildEMUReferenceHWModePipeline(mlir::OpPassManager& pm, const ReferenceHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // Level 3 : Topology

    pm.addPass(IE::createMatMulInputsTo2dPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    pm.addPass(IE::createUnrollBatchPass(log));

    if (options.enableConvertFCToConv) {
        pm.addPass(IE::createConvertFCToConvPass(log));
    }
    pm.addPass(IE::createConvertReduceToPoolingPass(log));
    pm.addPass(IE::createHandleLargeKernelsPass(log));
    pm.addPass(IE::createHandleExcludePadForAvgPoolPass(log));
    if (options.enableConvertAvgPoolToDWConv) {
        pm.addPass(IE::createConvertAvgPoolToDWConvPass(log));
    }

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
    }
    pm.addPass(IE::createFusePostOpsPass(log));

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);

    if (options.enableOptimizeReorders) {
        pm.addPass(IE::createOptimizeReordersPass(log));
        pm.addPass(IE::createUniquifyOpsPass(log));
    }

    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(createConvertIEToVPUNCEPass(log));
    pm.addPass(createConvertLayers2VPUPass(log));
    pm.addPass(VPU::createResolvePWLPostOpsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // TODO: revisit this pipeline definition after lowering of VPU.NCE ops to EMU.NCEClusterTask

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
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    pm.addPass(IE::createUnrollBatchPass(log));
    if (options.enableConvertFCToConv) {
        pm.addPass(IE::createConvertFCToConvPass(log));
    }
    pm.addPass(IE::createConvertReduceToPoolingPass(log));
    pm.addPass(IE::createConvertPowerToMultPass(log));

    if (options.enableHandleLargeKernel) {
        pm.addPass(IE::createHandleLargeKernelsPass(log));
    }
    pm.addPass(IE::createHandleExcludePadForAvgPoolPass(log));
    if (options.enableConvertAvgPoolToDWConv) {
        pm.addPass(IE::createConvertAvgPoolToDWConvPass(log));
    }

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
    }
    pm.addPass(IE::createFusePostOpsPass(log));
    pm.addPass(IE::createUnrollBatchPass(log));

    if (options.enableUpstreamSlice) {
        pm.addPass(IE::createUpstreamSlicePass(log));
    }

    IE::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);

    if (options.enableOptimizeReorders) {
        pm.addPass(IE::createOptimizeReordersPass(log));
        pm.addPass(IE::createUniquifyOpsPass(log));
    }

    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(createConvertIEToVPUNCEPass(log));
    pm.addPass(createConvertLayers2VPUPass(log));
    pm.addPass(VPU::createResolvePWLPostOpsPass(log));

    // TODO: revisit this pipeline definition after lowering of VPU.NCE ops to EMU.NCEClusterTask

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
