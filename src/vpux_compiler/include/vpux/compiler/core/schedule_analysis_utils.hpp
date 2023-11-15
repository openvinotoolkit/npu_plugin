//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/feasible_memory_scheduler.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/dpu_tiler.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/utils/plugin/profiling_json.hpp"

namespace vpux {

// Vector of stalls: pair {cycleStart, cycleEnd}
using StallCycles = SmallVector<std::pair<size_t, size_t>>;
// Map: Key: pair {executorKind, executorInstance}, Value: StallCycles
using ExecutorStallCycles = DenseMap<std::pair<VPU::ExecutorKind, size_t>, StallCycles>;
using ScheduledOpInfo = FeasibleMemoryScheduler::ScheduledOpInfo;
using ScheduledOpInfoVec = FeasibleMemoryScheduler::ScheduledOpInfoVec;

struct SpillStats {
    size_t numOfSpillWrites;
    size_t numOfSpillWritesDueToFrag;
    size_t numOfSpillWritesOfDataOps;
    size_t numOfSpillRead;
};

static const std::map<std::string, int> executorStrToId = {{"DMA_NN", 0},    {"DPU", 1},       {"NCE", 2},
                                                           {"SHAVE_UPA", 3}, {"SHAVE_ACT", 4}, {"SHAVE_NN", 5}};

ExecutorStallCycles getExecutorStallRegions(ScheduledOpInfoVec& scheduledOps);
StallCycles getStallsOnAllExecutorPipelines(ScheduledOpInfoVec& scheduledOps);
void verifyDependenciesPreservedInCycles(AsyncDepsInfo& depsInfo, ScheduledOpInfoVec& scheduledOps);
StringRef getTaskType(const ScheduledOpInfo& op);
void printScheduleStatistics(mlir::func::FuncOp& netFunc, AsyncDepsInfo& depsInfo, Logger log,
                             llvm::ArrayRef<ScheduledOpInfo> scheduledOps);
SpillStats getDynamicSpillingStats(llvm::ArrayRef<ScheduledOpInfo> scheduledOps);
void printSpillingStatistics(Logger log, SpillStats& beforePrefetching, SpillStats& afterPrefetching,
                             SpillStats& afterOptimizations);
void createTracingJSON(mlir::func::FuncOp& netFunc, StringRef fileName = "scheduleTrace.json");

}  // namespace vpux
