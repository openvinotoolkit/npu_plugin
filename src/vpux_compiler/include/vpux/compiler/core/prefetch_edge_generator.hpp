//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler.hpp"

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include <llvm/ADT/BitVector.h>

namespace vpux {

class PrefetchEdgeGenerator final {
    using ScheduledOpInfo = vpux::FeasibleMemoryScheduler::ScheduledOpInfo;
    using scheduledOps = llvm::SmallVector<ScheduledOpInfo>;
    using operationIdxType = vpux::FeasibleMemoryScheduler::operationIdxType;
    using prefetchMap = vpux::FeasibleMemoryScheduler::prefetchMap;

public:
    explicit PrefetchEdgeGenerator(scheduledOps& initialSchedule, AsyncDepsInfo& depsInfo);

public:
    prefetchMap generatePrefetchEdges();

private:
    bool prefetchConstraintsSatisifed(ScheduledOpInfo* dataOp, ScheduledOpInfo* computeOp,
                                      size_t currentComputeOpLevel);
    bool allDataOpDependenciesExecuted(operationIdxType dataIdx);
    bool canDataOpBePrefetched(ScheduledOpInfo* dataOp);
    bool isEltwiseOp(ScheduledOpInfo* op);

private:
    Logger _log;
    // incoming objects
    scheduledOps _scheduledOps;
    AsyncDepsInfo& _depsInfo;
    // class objects
    prefetchMap _prefetchEdges;
    std::unordered_set<operationIdxType> _prefetchedDataOps;
    std::unordered_set<operationIdxType> _executedOps;
    // prefetching constraints
    size_t PREFETCH_LEVEL_LIMIT_CONST = 2;
    size_t PREFETCH_LEVEL_LIMIT_ACT = 1;
    size_t PREFETCH_TIME_LIMIT = 50;
};

}  // namespace vpux
