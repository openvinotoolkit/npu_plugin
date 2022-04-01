//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler.hpp"
#include "vpux/compiler/core/linear_scan_handler.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace vpux {

//
// FeasibleMemorySchedulerSpilling class is a support class for FeasibleMemoryScheduler
// to handle spilling and insert correct spill-write and spill-read CopyOps within
// async dialect and perform required connections between new ops to achieve correct
// execution order for models that need spilling
//
class FeasibleMemorySchedulerSpilling final {
public:
    explicit FeasibleMemorySchedulerSpilling(mlir::FuncOp netFunc, VPU::MemoryKind memKind,
                                             Optional<VPU::MemoryKind> secondLvlMemKind, AsyncDepsInfo& depsInfo,
                                             AliasesInfo& aliasInfo, Logger log,
                                             LinearScan<mlir::Value, LinearScanHandler>& scan);

    void removeComputeOpRelocationSpills(SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo>& scheduledOps);
    void optimizeDataOpsSpills(SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo>& scheduledOps);
    void removeRedundantSpillWrites(SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo>& scheduledOps);
    void insertSpillCopyOps(SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo>& scheduledOps);

private:
    void createSpillWrite(SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo>& scheduledOps, size_t schedOpIndex);
    void createSpillRead(SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo>& scheduledOps, size_t schedOpIndex);
    mlir::async::ExecuteOp insertSpillWriteCopyOp(mlir::async::ExecuteOp opThatWasSpilled,
                                                  mlir::async::ExecuteOp insertAfterExecOp, mlir::Value bufferToSpill,
                                                  size_t allocatedAddress);
    mlir::async::ExecuteOp insertSpillReadCopyOp(mlir::async::ExecuteOp opThatWasSpilled, mlir::Value bufferToSpill,
                                                 mlir::async::ExecuteOp spillWriteExecOp,
                                                 mlir::async::ExecuteOp insertAfterExecOp, size_t allocatedAddress);
    void updateSpillWriteReadUsers(mlir::Value bufferToSpill, mlir::async::ExecuteOp spillWriteExecOp,
                                   mlir::async::ExecuteOp spillReadExecOp);
    SmallVector<mlir::Value> getAsyncResultsForBuffer(mlir::async::ExecuteOp opThatWasSpilled, mlir::Value buffer);
    mlir::Value getBufferFromAsyncResult(mlir::Value asyncResult);

    // Below nested class is intended to handle data dependency updates
    // for users of spilled buffers
    class SpillUsersUpdate {
    public:
        explicit SpillUsersUpdate(FeasibleMemorySchedulerSpilling& spillingClass,
                                  mlir::async::ExecuteOp opThatWasSpilled, mlir::async::ExecuteOp spillReadExecOp,
                                  mlir::Value bufferToSpill)
                : _spillingParentClass(spillingClass),
                  _opThatWasSpilled(opThatWasSpilled),
                  _spillReadExecOp(spillReadExecOp),
                  _bufferToSpill(bufferToSpill) {
        }
        void resolveSpillBufferUsage();

    private:
        mlir::Operation* getViewOpForMasterBuffer(mlir::Value asyncResult);
        SmallVector<mlir::async::ExecuteOp> getUsersOfSpilledOpThatNeedUpdate(mlir::Value opThatWasSpilledResult);
        unsigned int getOperandIndexForSpillResultUser(mlir::async::ExecuteOp spillResultUser,
                                                       mlir::Value spilledAsyncResult);
        void updateSpillResultUsers(mlir::Value oldResult, mlir::Value newResult);
        void updateSpillBufferUsers(mlir::Value oldBuffer, mlir::Value newBuffer);

        FeasibleMemorySchedulerSpilling& _spillingParentClass;
        mlir::async::ExecuteOp _opThatWasSpilled;
        mlir::async::ExecuteOp _spillReadExecOp;
        mlir::Value _bufferToSpill;
    };

private:
    Logger _log;
    // first level mem space
    VPU::MemoryKind _memKind;
    // second level mem space which is used for spilling
    mlir::Optional<VPU::MemoryKind> _secondLvlMemKind;
    // dependencies of ops
    AsyncDepsInfo& _depsInfo;
    // aliases information for buffers
    AliasesInfo& _aliasInfo;
    // allocator class
    LinearScan<mlir::Value, LinearScanHandler>& _scan;
    // insertion point for allocation related operations
    mlir::Operation* _allocOpInsertionPoint;
    // Vector of pairs of operation ID and inserted spill-write exec-op that doesn't have yet corresponding spill-read
    // op
    SmallVector<std::pair<mlir::Value, mlir::async::ExecuteOp>> _opIdAndSpillWritePairs;
    // Map storing new buffers replacing spilled buffers: key - original spilled buffer, value - new allocated buffer
    // after spill-read
    llvm::DenseMap<mlir::Value, mlir::Value> _bufferReplacementAfterSpillRead;
};

}  // namespace vpux
