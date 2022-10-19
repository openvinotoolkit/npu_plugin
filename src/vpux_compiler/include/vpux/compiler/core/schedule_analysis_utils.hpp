//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/feasible_memory_scheduler.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/dpu_tiler.hpp"
#include "vpux/compiler/utils/strings.hpp"
#include "vpux/utils/plugin/profiling_json.hpp"

namespace vpux {

using StallCycles = SmallVector<std::pair<size_t, size_t>>;
using ExecutorStallCycles = DenseMap<VPU::ExecutorKind, StallCycles>;
using ScheduledOpInfo = FeasibleMemoryScheduler::ScheduledOpInfo;

static const std::map<std::string, int> executorStrToId = {{"DMA_NN", 0},    {"DPU", 1},       {"NCE", 2},
                                                           {"SHAVE_UPA", 3}, {"SHAVE_ACT", 4}, {"SHAVE_NN", 5}};

ExecutorStallCycles getExecutorStallRegions(SmallVector<ScheduledOpInfo>& scheduledOps);
StallCycles getStallsOnAllExecutorPipelines(SmallVector<ScheduledOpInfo>& scheduledOps);
void verifyDependenciesPreservedInCycles(AsyncDepsInfo& depsInfo, SmallVector<ScheduledOpInfo>& scheduledOps);
StringRef getTaskType(ScheduledOpInfo op);
void printScheduleStatistics(mlir::FuncOp& netFunc, AsyncDepsInfo& depsInfo, Logger log,
                             llvm::ArrayRef<ScheduledOpInfo> scheduledOps);
void createTracingJSON(mlir::FuncOp& netFunc, StringRef fileName = "scheduleTrace.json");

}  // namespace vpux
