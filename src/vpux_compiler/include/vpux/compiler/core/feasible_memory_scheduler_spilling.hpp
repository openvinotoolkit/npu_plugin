//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
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
    explicit FeasibleMemorySchedulerSpilling(mlir::FuncOp netFunc, IndexedSymbolAttr memSpace,
                                             IndexedSymbolAttr secondLvlMemSpace, AsyncDepsInfo& depsInfo,
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
    IndexedSymbolAttr _memSpace;
    // second level mem space which is used for spilling
    IndexedSymbolAttr _secondLvlMemSpace;
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
