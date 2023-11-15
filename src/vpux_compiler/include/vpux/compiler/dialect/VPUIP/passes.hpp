//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/optional.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>

namespace vpux {
namespace VPUIP {

//
// Passes
//

using MemKindCreateFunc = std::function<Optional<VPU::MemoryKind>(StringRef)>;
using ConditionFunc = std::function<bool(mlir::Operation*)>;

template <typename T>
bool isOp(mlir::Operation* op) {
    return mlir::isa<T>(op);
}

static ConditionFunc makeStubCondition() {
    return &vpux::VPUIP::isOp<IE::MemPermuteOp>;
}

std::unique_ptr<mlir::Pass> createConvertWeightsTableOp2ConstPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDumpStatisticsOfTaskOpsPass(bool enableCompressWeightsBtc = true,
                                                              Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollClusterTilingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnwrapClusterTilingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollDepthToSpaceDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollSpaceToDepthDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollUpsamplingDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollPermuteToNNDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollExpandDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollPerAxisTileDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollSwKernelPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDMABarrierOptimizationPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseConstantsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPatchFusedConstantsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createResolveDMAWithSwizzlingPass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createMovePureViewOpBeforeCopyPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeCopiesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeConcatViewCopiesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCopyOpHoistingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeParallelCopiesPass(bool enableOptimizeConstCopy = true,
                                                             Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMoveSubViewBeforeSparseBufferPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCopyOpTilingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSetMemorySpacePass(MemKindCreateFunc memKindCb, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createStaticAllocationPass(MemKindCreateFunc memKindCb, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertEltwiseToInPlacePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLinearizationPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFeasibleAllocationPass(MemKindCreateFunc memKindCb,
                                                         MemKindCreateFunc secondLvlMemKindCb = nullptr,
                                                         Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMaximizeUPACyclesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createBreakDataFlowPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPatchWeightsTablePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDMATaskProfilingReserveMemPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDMATaskProfilingAfterBarrierSchedPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCaptureWorkpointPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDPUProfilingPass(MemKindCreateFunc memKindCb, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUPAProfilingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createGroupProfilingBuffersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createActShaveProfilingPass(MemKindCreateFunc memKindCb, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWrapWithPermuteAsNNDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertExpandPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwizzlingPass(const bool enableWeightSwizzling = true,
                                                const bool enableActivationSwizzling = true,
                                                Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOperationStubbingPass(ConditionFunc condition = makeStubCondition(),
                                                        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvWeightsCompressionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUngroupSparseBuffersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateCompressionSchemePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFlattenSparseWeightsTypesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createComputeSESizesPass(Optional<bool> onlyInputsConcatOverC = None,
                                                     Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createComputeSEBasePtrsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertSETablesToConstantsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustInputDataForExplicitSETablePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createTileActShaveKernelTaskPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustSpillSizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCompressDmaReserveMemPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseDDRCopiesIntoConcats(Logger log = Logger::global());

//
// Asynchronous Scheduling pipeline
//

void buildAsyncSchedulingPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createWrapIntoAsyncRegionsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMoveWaitResultToAsyncBlockArgsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCalculateAsyncRegionCycleCostPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createGroupAsyncExecuteOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMoveViewOpsIntoAsyncRegionsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeAsyncDepsPass(Logger log = Logger::global());

//
// Hardware Adaptation pipeline
//

void buildHardwareAdaptationPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createConvertTransferOpsToDMAsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertAllocationsToDeclarationsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertFuncArgsToDeclarationsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertViewOpsToDeclarationsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertAsyncOpsToTasksPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCompressWeightsBTCPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createNNDMATilingPass(Logger log = Logger::global());

//
// Registration
//

void registerVPUIPPipelines();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/dialect/VPUIP/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/dialect/VPUIP/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace VPUIP
}  // namespace vpux
