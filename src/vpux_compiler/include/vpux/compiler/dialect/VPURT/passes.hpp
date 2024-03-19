//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>

namespace vpux {
namespace VPURT {

//
// Barrier Legalization options
//

struct BarrierLegalizationOptions : mlir::PassPipelineOptions<BarrierLegalizationOptions> {
    BoolOption simpleSchedule{*this, "simple-schedule", llvm::cl::desc("Simplified schedule"), llvm::cl::init(false)};
    BoolOption shareWaitAndUpdateBarriers{*this, "share-wait-and-update-barriers",
                                          llvm::cl::desc("Share wait and update barriers"), llvm::cl::init(true)};
    BoolOption reduceParallelControlFlows{*this, "reduce-parallel-control-flows",
                                          llvm::cl::desc("Reduce parallel overlapping control flows where possible"),
                                          llvm::cl::init(true)};

    BarrierLegalizationOptions() = default;

    template <class OtherOptions>
    explicit BarrierLegalizationOptions(const OtherOptions& options) {
        simpleSchedule = options.enableSimpleSchedule;
        shareWaitAndUpdateBarriers = options.shareWaitAndUpdateBarriers;
        reduceParallelControlFlows = options.reduceParallelControlFlows;
    }
};

//
// Barrier Legalization Pipeline
//

void buildBarrierLegalizationPipeline(mlir::OpPassManager& pm, const VPURT::BarrierLegalizationOptions& options,
                                      Logger log = Logger::global());

//
// Passes
//

std::unique_ptr<mlir::Pass> createSimplifySchedulePass(const bool shareWaitAndUpdateBarriersFlag = true,
                                                       const bool reduceParallelControlFlowsFlag = true,
                                                       Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitExceedingVariantCountBarriersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSatisfyOneWaitBarrierPerTaskPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createReduceExceedingActiveCountBarriersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAssignPhysicalBarriersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createBarrierSimulationPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createInferenceExecutionAnalysisPass(
        std::string compileSchedTraceFileName = "compileTimeScheduleTrace.json", bool dumpToJson = false,
        bool enableActivityFactor = true, Logger log = Logger::global());

//
// Registration
//

void registerVPURTPipelines();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/dialect/VPURT/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/dialect/VPURT/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace VPURT
}  // namespace vpux
