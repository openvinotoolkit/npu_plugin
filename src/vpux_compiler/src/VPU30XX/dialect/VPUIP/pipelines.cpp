//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU30XX/dialect/VPUIP/passes.hpp"

#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/dialect/VPURT/passes.hpp"
#include "vpux/compiler/dialect/const/passes.hpp"

#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

void vpux::VPUIP::arch30xx::buildMemoryAllocationPipeline(mlir::OpPassManager& pm,
                                                          const VPUIP::arch30xx::MemoryAllocationOptions& options,
                                                          Logger log) {
    pm.addPass(VPUIP::createFeasibleAllocationPass(
            VPU::getMemKind<VPU::MemoryKind::CMX_NN>, VPU::getMemKind<VPU::MemoryKind::DDR>, options.linearizeSchedule,
            options.enablePipelining, options.enablePrefetching, options.optimizeFragmentation,
            options.optimizeDynamicSpilling, log));

    pm.addPass(VPUIP::createMaximizeUPACyclesPass(log));

    if (options.enableGroupAsyncExecuteOps) {
        pm.addPass(VPUIP::createGroupAsyncExecuteOpsPass(log));
    }

    pm.addPass(VPUIP::createQueryArgsAllocationAnalysisPass());
    pm.addPass(VPUIP::createStaticAllocationPass(VPU::getMemKind<VPU::MemoryKind::DDR>, log));
    pm.addPass(VPUIP::createCollectUsedMemoryPass());
}

void vpux::VPUIP::arch30xx::buildDefaultHWPipeline(mlir::OpPassManager& pm,
                                                   const VPUIP::arch30xx::DefaultHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(VPUIP::createConvertExpandPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPUIP::createConvertEltwiseToInPlacePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // Level 2 : Abstract RunTime

    pm.addPass(VPUIP::createSetMemorySpacePass(VPU::getMemKind<VPU::MemoryKind::DDR>, log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableOptimizeCopies) {
        pm.addPass(VPUIP::createMovePureViewOpBeforeCopyPass(log));
        pm.addPass(VPUIP::createOptimizeCopiesPass(log));
        pm.addPass(VPUIP::createOptimizeConcatViewCopiesPass(log));
        pm.addPass(VPUIP::createFuseDDRCopiesIntoConcats(log));
        pm.addPass(VPUIP::createOptimizeParallelCopiesPass(options.enableOptimizeConstCopies, log));
        pm.addPass(VPUIP::createMovePureViewOpBeforeCopyPass(log));
    }

    pm.addPass(VPUIP::createConvertToDMAPass(log));
    pm.addPass(VPUIP::createCopyOpTilingPass(log));

    if (options.enableSEPtrsOperations || options.enableSEPTransposedConv) {
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
        pm.addPass(VPUIP::createDPUProfilingPass(VPU::getMemKind<VPU::MemoryKind::CMX_NN>, log));
    }
    if (options.enableProfiling && options.enableSWProfiling) {
        pm.addPass(VPUIP::createActShaveProfilingPass(VPU::getMemKind<VPU::MemoryKind::CMX_NN>, log));
    }

    pm.addPass(VPUIP::createConvertTransferOpsToDMAsPass(log));

    VPUIP::buildAsyncSchedulingPipeline(pm, log);

    if (options.enableProfiling && options.enableDMAProfiling) {
        pm.addPass(VPUIP::createDMATaskProfilingReserveMemPass(log));
    }

    pm.addPass(VPUIP::createCalculateAsyncRegionCycleCostPass(log));

    VPUIP::arch30xx::buildMemoryAllocationPipeline(pm, VPUIP::arch30xx::MemoryAllocationOptions(options), log);

    pm.addPass(VPUIP::createOptimizeAsyncDepsPass(log));
    pm.addPass(VPUIP::createBreakDataFlowPass(log));

    pm.addPass(VPUIP::createUnwrapClusterTilingPass(log));

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

    if (!options.linearizeSchedule) {
        pm.addPass(VPUIP::createDMABarrierOptimizationPass(log));
    }

    VPURT::buildBarrierLegalizationPipeline(pm, VPURT::BarrierLegalizationOptions(options), log);

    if (options.enableWeightsSparsity) {
        pm.addPass(VPUIP::createFlattenSparseWeightsTypesPass(log));
    }
    if (options.enableSEPtrsOperations || options.enableSEPTransposedConv) {
        pm.addPass(VPUIP::createComputeSESizesPass(/*onlyInputsConcatOverC=*/false, log));
        pm.addPass(VPUIP::createAdjustInputDataForExplicitSETablePass(log));
    }

    pm.addPass(VPUIP::createUnrollDepthToSpaceDMAPass(log));
    pm.addPass(VPUIP::createUnrollSpaceToDepthDMAPass(log));
    pm.addPass(VPUIP::createUnrollPermuteToNNDMAPass(log));

    pm.addPass(VPUIP::createUnrollUpsamplingDMAPass(log));
    pm.addPass(VPUIP::createUnrollExpandDMAPass(log));
    pm.addPass(VPUIP::createUnrollPerAxisTileDMAPass(log));

    if (!options.linearizeSchedule) {
        pm.addPass(VPUIP::createDMABarrierOptimizationPass(log));
    }

    if (options.enableProfiling) {
        if (options.enableDMAProfiling) {
            pm.addPass(VPUIP::createDMATaskProfilingAfterBarrierSchedPass(log));
        }
        pm.addPass(VPUIP::createGroupProfilingBuffersPass(log));
        pm.addPass(createMoveDeclarationsToTopPass(log));
    }

    pm.addPass(VPURT::createAssignPhysicalBarriersPass(log));
    pm.addPass(VPURT::createBarrierSimulationPass(log));
    pm.addPass(Const::createConstantFoldingPass());

    if (options.enableScheduleTrace) {
        pm.addPass(VPURT::createInferenceExecutionAnalysisPass(options.scheduleTraceFile, options.enableScheduleTrace,
                                                               false, log));
    }
    if (options.enableDumpTaskStats) {
        // Force logging if dump-task-stats was enabled explicitly on the command line
        pm.addPass(VPUIP::createDumpStatisticsOfTaskOpsPass(
                log, options.enableDumpTaskStats.hasValue() && options.enableDumpTaskStats));
    }
}

void vpux::VPUIP::arch30xx::registerVPUIPPipelines() {
    mlir::PassPipelineRegistration<VPUIP::arch30xx::MemoryAllocationOptions>(
            "memory-allocation", "Memory Allocation",
            [](mlir::OpPassManager& pm, const VPUIP::arch30xx::MemoryAllocationOptions& options) {
                VPUIP::arch30xx::buildMemoryAllocationPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<VPUIP::arch30xx::DefaultHWOptions>(
            "default-hw-mode-vpuip", "VPUIP dialect part of Default HW pipeline",
            [](mlir::OpPassManager& pm, const VPUIP::arch30xx::DefaultHWOptions& options) {
                VPUIP::arch30xx::buildDefaultHWPipeline(pm, options);
            });
}
