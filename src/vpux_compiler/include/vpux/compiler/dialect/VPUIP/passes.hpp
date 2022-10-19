//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

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

std::unique_ptr<mlir::Pass> createConvertWeightsTableOp2ConstPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDumpStatisticsOfTaskOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollClusterTilingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollDepthToSpaceDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollPermuteToNNDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDMABarrierOptimizationPass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createOptimizeCopiesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCopyOpHoistingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeParallelCopiesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCopyOpTilingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSetInternalMemorySpacePass(MemKindCreateFunc memKindCb,
                                                             Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createStaticAllocationPass(MemKindCreateFunc memKindCb, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLinearizationPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFeasibleAllocationPass(MemKindCreateFunc memKindCb,
                                                         MemKindCreateFunc secondLvlMemKindCb = nullptr,
                                                         Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMaximizeUPACyclesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createBreakDataFlowPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPatchWeightsTablePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwizzleConstantPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDMATaskProfilingPass(MemKindCreateFunc memKindCb, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDPUProfilingPass(MemKindCreateFunc memKindCb, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUPAProfilingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createGroupProfilingBuffersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createActShaveProfilingPass(MemKindCreateFunc memKindCb, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMoveOperationFromDDRtoCMXPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAlignmentForSwizzling(bool enableWeightSwizzling = true,
                                                        bool enableActivationSwizzling = true,
                                                        Logger log = Logger::global());

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

struct HardwareAdaptationOptions : mlir::PassPipelineOptions<HardwareAdaptationOptions> {
    BoolOption enableCompressWeights{*this, "compress-weights", llvm::cl::desc("Enable compress-weights pass"),
                                     llvm::cl::init(false)};

    HardwareAdaptationOptions() = default;

    template <
            class OtherOptions,
            typename = std::enable_if_t<std::is_base_of<mlir::PassPipelineOptions<OtherOptions>, OtherOptions>::value>>
    explicit HardwareAdaptationOptions(const OtherOptions& options) {
        enableCompressWeights = options.enableCompressWeights;
    }
};

void buildHardwareAdaptationPipeline(mlir::OpPassManager& pm, const HardwareAdaptationOptions& options,
                                     Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createConvertTransferOpsToDMAsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertAllocationsToDeclarationsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertViewOpsToDeclarationsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertAsyncOpsToTasksPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCompressWeightsPass(Logger log = Logger::global());

//
// Registration
//

void registerVPUIPPipelines();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/dialect/VPUIP/generated/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/dialect/VPUIP/generated/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace VPUIP
}  // namespace vpux
