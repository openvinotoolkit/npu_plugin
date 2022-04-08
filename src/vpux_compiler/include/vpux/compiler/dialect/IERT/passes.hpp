//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <functional>
#include <memory>

namespace vpux {
namespace IERT {

//
// Passes
//

using MemKindCreateFunc = std::function<Optional<VPU::MemoryKind>(StringRef)>;

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
std::unique_ptr<mlir::Pass> createBreakDataFlowPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPatchWeightsTablePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDMATaskProfilingPass(MemKindCreateFunc memKindCb, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDPUProfilingPass(MemKindCreateFunc memKindCb, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertScalarToTensorPass(Logger log = Logger::global());
//
// Asynchronous Scheduling pipeline
//

void buildAsyncSchedulingPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createWrapIntoAsyncRegionsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMoveWaitResultToAsyncBlockArgsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createGroupAsyncExecuteOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMoveViewOpsIntoAsyncRegionsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeAsyncDepsPass(Logger log = Logger::global());

//
// Registration
//

void registerIERTPipelines();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/dialect/IERT/generated/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/dialect/IERT/generated/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace IERT
}  // namespace vpux
