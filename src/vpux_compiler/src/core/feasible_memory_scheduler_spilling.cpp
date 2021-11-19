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

#include "vpux/compiler/core/feasible_memory_scheduler_spilling.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

//
// Feasible Memory Scheduler Spilling support
//

FeasibleMemorySchedulerSpilling::FeasibleMemorySchedulerSpilling(mlir::FuncOp netFunc, mlir::Attribute memSpace,
                                                                 mlir::Attribute secondLvlMemSpace,
                                                                 AsyncDepsInfo& depsInfo, AliasesInfo& aliasInfo,
                                                                 Logger log,
                                                                 LinearScan<mlir::Value, LinearScanHandler>& scan)
        : _log(log),
          _memSpace(memSpace),
          _secondLvlMemSpace(secondLvlMemSpace),
          _depsInfo(depsInfo),
          _aliasInfo(aliasInfo),
          _scan(scan) {
    _log.setName("feasible-memory-scheduler-spilling");
    // Locate first async-exec-op that will be used to determine insertion point for
    // new allocation operations
    _allocOpInsertionPoint = *(netFunc.getOps<mlir::async::ExecuteOp>().begin());
    VPUX_THROW_UNLESS(_allocOpInsertionPoint != nullptr,
                      "Unable to find insertion point for new allocation operations");
}

mlir::Value FeasibleMemorySchedulerSpilling::getAsyncResultForBuffer(mlir::Value buffer) {
    for (auto bufferAlias : _aliasInfo.getAllAliases(buffer)) {
        if (bufferAlias.getType().isa<mlir::async::ValueType>()) {
            return bufferAlias;
        }
    }
    VPUX_THROW("No async result matched for a given buffer");
}

mlir::Value FeasibleMemorySchedulerSpilling::getBufferFromAsyncResult(mlir::Value asyncResult) {
    const auto resultType = asyncResult.getType();
    VPUX_THROW_UNLESS(resultType.isa<mlir::async::ValueType>(), "This is not async result. Got: '{0}'", resultType);
    return _aliasInfo.getRoot(asyncResult);
}

mlir::async::ExecuteOp FeasibleMemorySchedulerSpilling::insertSpillWriteCopyOp(mlir::async::ExecuteOp opThatWasSpilled,
                                                                               mlir::async::ExecuteOp insertAfterExecOp,
                                                                               mlir::Value bufferToSpill,
                                                                               size_t allocatedAddress) {
    auto spillWriteNameLoc = appendLoc(opThatWasSpilled->getLoc(),
                                       llvm::formatv("spill_write_{0}", _depsInfo.getIndex(opThatWasSpilled)).str());
    _log.trace("Insert Spill Write copyOp - '{0}'", spillWriteNameLoc);

    // Get information about current returned memref type and prepare new one with proper memory location
    auto opToSpillResult = getAsyncResultForBuffer(bufferToSpill);
    auto opToSpillAsyncType = opToSpillResult.getType().dyn_cast<mlir::async::ValueType>();
    auto opToSpillMemRefType = opToSpillAsyncType.getValueType().cast<mlir::MemRefType>();
    auto spillBufferMemType = changeMemSpace(opToSpillMemRefType, _secondLvlMemSpace);

    // Update address of the buffer that is to be spilled as spillWrite source buffer
    // is not correctly configured during scheduler memory allocation
    _scan.handler().setAddress(bufferToSpill, allocatedAddress);

    // Create buffer in second level memory
    mlir::OpBuilder builder(_allocOpInsertionPoint);
    builder.setInsertionPoint(_allocOpInsertionPoint);
    auto spillBuffer = builder.create<mlir::memref::AllocOp>(spillWriteNameLoc, spillBufferMemType);

    // Create new AsyncExecOp
    builder.setInsertionPointAfter(insertAfterExecOp);
    auto spillWriteExecOp =
            builder.create<mlir::async::ExecuteOp>(spillWriteNameLoc, spillBuffer->getResultTypes(), None, None);

    // Update operands of new AsyncExecOp to contain result of AsyncExecOp to be spilled
    spillWriteExecOp.operandsMutable().append(opToSpillResult);
    auto innerArg = spillWriteExecOp.getBody()->addArgument(opToSpillAsyncType.getValueType());

    // Create CopyOp in the body of new AsyncExecOp
    auto bodyBlock = &spillWriteExecOp.body().front();
    builder.setInsertionPointToStart(bodyBlock);
    auto spillWriteCopyOp = builder.create<IERT::CopyOp>(spillWriteNameLoc, innerArg, spillBuffer.memref());
    builder.create<mlir::async::YieldOp>(spillWriteNameLoc, spillWriteCopyOp->getResults());

    // Add token dependencies
    spillWriteExecOp.dependenciesMutable().assign(makeArrayRef(opThatWasSpilled.token()));

    // Update aliases for spillWrite result
    _aliasInfo.addAlias(spillBuffer.memref(), spillBuffer.memref());
    _aliasInfo.addAlias(spillBuffer.memref(), spillWriteExecOp.results()[0]);
    _aliasInfo.addAlias(spillBuffer.memref(), spillWriteCopyOp.output());

    // Update executor attributes of new AsyncExecOp
    uint32_t numExecutorUnits = 0;
    auto copyOpExecutor = mlir::dyn_cast_or_null<IERT::AsyncLayerOpInterface>(spillWriteCopyOp.getOperation());
    auto executor = copyOpExecutor.getExecutor(numExecutorUnits);
    if (executor != nullptr) {
        IERT::IERTDialect::setExecutor(spillWriteExecOp, executor, numExecutorUnits);
    }

    // Update dependencies map and get new operation index
    _depsInfo.insertNewExecOpToDepsMap(spillWriteExecOp);

    return spillWriteExecOp;
}

mlir::async::ExecuteOp FeasibleMemorySchedulerSpilling::insertSpillReadCopyOp(mlir::async::ExecuteOp opThatWasSpilled,
                                                                              mlir::async::ExecuteOp spillWriteExecOp,
                                                                              mlir::async::ExecuteOp insertAfterExecOp,
                                                                              size_t allocatedAddress) {
    auto spillReadNameLoc = appendLoc(opThatWasSpilled->getLoc(),
                                      llvm::formatv("spill_read_{0}", _depsInfo.getIndex(opThatWasSpilled)).str());
    _log.trace("Insert Spill Read copyOp - '{0}'", spillReadNameLoc);

    // Get information about spill write returned memref type and prepare new one with proper memory location
    auto spillWriteResult = spillWriteExecOp.results()[0];
    auto spillWriteAsyncType = spillWriteResult.getType().dyn_cast<mlir::async::ValueType>();
    auto spillWriteMemRefType = spillWriteAsyncType.getValueType().cast<mlir::MemRefType>();
    auto newBufferMemType = changeMemSpace(spillWriteMemRefType, _memSpace);

    // Create buffer in first level memory to bring back spilled buffer. Configure its
    // address as since it is a new buffer corresponding AllocOp operation was not set
    // an address
    mlir::OpBuilder builder(_allocOpInsertionPoint);
    builder.setInsertionPoint(_allocOpInsertionPoint);
    auto newBuffer = builder.create<mlir::memref::AllocOp>(spillReadNameLoc, newBufferMemType);
    _scan.handler().setAddress(newBuffer.memref(), allocatedAddress);

    // Create new AsyncExecOp in correct place
    builder.setInsertionPointAfter(insertAfterExecOp);
    auto spillReadExecOp =
            builder.create<mlir::async::ExecuteOp>(spillReadNameLoc, newBuffer->getResultTypes(), None, None);

    // Update operands of new AsyncExecOp to contain result of AsyncExecOp of spillWrite
    spillReadExecOp.operandsMutable().append(spillWriteResult);
    auto innerArg = spillReadExecOp.getBody()->addArgument(spillWriteAsyncType.getValueType());

    // Create CopyOp in the body of new AsyncExecOp
    auto bodyBlock = &spillReadExecOp.body().front();
    builder.setInsertionPointToStart(bodyBlock);
    auto spillReadCopyOp = builder.create<IERT::CopyOp>(spillReadNameLoc, innerArg, newBuffer.memref());
    builder.create<mlir::async::YieldOp>(spillReadNameLoc, spillReadCopyOp->getResults());

    // Add token dependencies
    spillReadExecOp.dependenciesMutable().assign(makeArrayRef(spillWriteExecOp.token()));

    // Update alias for spillRead result
    _aliasInfo.addAlias(newBuffer.memref(), newBuffer.memref());
    _aliasInfo.addAlias(newBuffer.memref(), spillReadExecOp.results()[0]);
    _aliasInfo.addAlias(newBuffer.memref(), spillReadCopyOp.output());

    // Update executor attributes of new AsyncExecOp
    uint32_t numExecutorUnits = 0;
    auto copyOpExecutor = mlir::dyn_cast_or_null<IERT::AsyncLayerOpInterface>(spillReadCopyOp.getOperation());
    auto executor = copyOpExecutor.getExecutor(numExecutorUnits);
    if (executor != nullptr) {
        IERT::IERTDialect::setExecutor(spillReadExecOp, executor, numExecutorUnits);
    }

    // Update dependencies map and get new operation index
    _depsInfo.insertNewExecOpToDepsMap(spillReadExecOp);

    return spillReadExecOp;
}

// This function will update operands of users of spilled buffer
// and make proper connections
void FeasibleMemorySchedulerSpilling::updateSpillWriteReadUsers(mlir::Value bufferToSpill,
                                                                mlir::async::ExecuteOp spillWriteExecOp,
                                                                mlir::async::ExecuteOp spillReadExecOp) {
    _log.trace("Update users of Spill Write-Read pair: '{0}' -> '{1}'", spillWriteExecOp->getLoc(),
               spillReadExecOp->getLoc());

    auto opThatWasSpilledResult = getAsyncResultForBuffer(bufferToSpill);
    auto spillReadExecOpResult = spillReadExecOp.results()[0];

    // Find operations which should be excluded from operand update to result of spillRead.
    // By default this is always spillWrite operation
    llvm::SmallPtrSet<mlir::Operation*, 1> excludedUsersFromOperandsUpdate = {spillWriteExecOp.getOperation()};
    for (auto* user : opThatWasSpilledResult.getUsers()) {
        if (mlir::isa_and_nonnull<mlir::async::ExecuteOp>(user) &&
            user->isBeforeInBlock(spillReadExecOp.getOperation())) {
            excludedUsersFromOperandsUpdate.insert(user);
        }
    }

    // Update connections opThatWasSpilled -> SpillWrite -> SpillRead -> UserOfSpilledBuffer
    opThatWasSpilledResult.replaceAllUsesExcept(spillReadExecOpResult, excludedUsersFromOperandsUpdate);

    // Add tokens matching those new data dependencies
    for (auto* user : spillReadExecOpResult.getUsers()) {
        if (auto userAsyncOp = mlir::dyn_cast_or_null<mlir::async::ExecuteOp>(user)) {
            userAsyncOp.dependenciesMutable().append(makeArrayRef(spillReadExecOp.token()));
        }
    }

    // If there are operations which were referring directly to output buffer that was spilled
    // they should be updated to refer to result of spillRead if they appear in the IR
    // after the op whose result was spilled
    // Get information about the users of original output buffer that should still refer to it
    // (e.g. operations that appear in IR before)
    llvm::SmallPtrSet<mlir::Operation*, 1> excludedUsersFromOrigBufferUpdate;
    for (auto* user : bufferToSpill.getUsers()) {
        if (user != nullptr) {
            if (mlir::isa_and_nonnull<mlir::async::ExecuteOp>(user->getParentOp()) &&
                user->getParentOp()->isBeforeInBlock(spillReadExecOp.getOperation())) {
                excludedUsersFromOrigBufferUpdate.insert(user);
            }
        }
    }

    // Get new output buffer that is the result of spillRead
    auto newOutputBuffer = getBufferFromAsyncResult(spillReadExecOpResult);

    // Update all users of original output buffer with the new buffer from spillRead except
    // the operations which were identified to refer to old output buffer
    bufferToSpill.replaceAllUsesExcept(newOutputBuffer, excludedUsersFromOrigBufferUpdate);
}

// Create Spill Write operation based on data from feasible scheduler
void FeasibleMemorySchedulerSpilling::createSpillWrite(
        llvm::SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo>& scheduledOps, size_t schedOpIndex) {
    auto& schedOp = scheduledOps[schedOpIndex];
    auto schedOpBuffer = schedOp.getBuffer(0);
    _log = _log.nest();
    _log.trace("Create Spill Write for buffer - '{0}'", schedOpBuffer);

    // Get the insertion point. Pick first non-implicit previous op
    // SpillWrite operation will be inserted just after it
    mlir::async::ExecuteOp spillWriteInsertionPoint = nullptr;
    auto insertionPointIndex = schedOpIndex;
    while (insertionPointIndex > 0) {
        if (scheduledOps[insertionPointIndex].opType_ == FeasibleMemoryScheduler::EOpType::ORIGINAL_OP) {
            spillWriteInsertionPoint = _depsInfo.getExecuteOpAtIndex(scheduledOps[insertionPointIndex].op_);
            break;
        }
        insertionPointIndex--;
    }
    VPUX_THROW_UNLESS(spillWriteInsertionPoint != nullptr, "No location to insert Spill Write was identified");

    // In scheduledOpInfo structure op_ identifier for a spillWrite operation contains id
    // of the original operation which result had to be spilled
    auto opThatWasSpilled = _depsInfo.getExecuteOpAtIndex(schedOp.op_);

    auto spillWriteExecOp =
            insertSpillWriteCopyOp(opThatWasSpilled, spillWriteInsertionPoint, schedOpBuffer, schedOp.beginResource(0));
    _opIdAndSpillWritePairs.push_back({schedOpBuffer, spillWriteExecOp});

    size_t spillWriteIndex = _depsInfo.getIndex(spillWriteExecOp);
    _log.trace("Spill Write new opId - '{0}'", spillWriteIndex);

    // After implicit spill write operation has been replaced with a proper copy op task then update
    // scheduled ops structure
    schedOp.opType_ = FeasibleMemoryScheduler::EOpType::ORIGINAL_OP;
    schedOp.op_ = spillWriteIndex;
    schedOp.resourceInfo_[0].invalidate();
    _log = _log.unnest();
}

// Create Spill Read operation based on data from feasible scheduler
void FeasibleMemorySchedulerSpilling::createSpillRead(
        llvm::SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo>& scheduledOps, size_t schedOpIndex) {
    auto& schedOp = scheduledOps[schedOpIndex];
    auto schedOpBuffer = schedOp.getBuffer(0);
    _log = _log.nest();
    _log.trace("Create Spill Read for buffer - '{0}'", schedOpBuffer);
    // Get spillWrite operation for the given spillRead to properly
    // connect both operations
    auto opIdAndSpillWritePair = std::find_if(_opIdAndSpillWritePairs.begin(), _opIdAndSpillWritePairs.end(),
                                              [&](std::pair<mlir::Value, mlir::async::ExecuteOp> pairElemenet) {
                                                  return (pairElemenet.first == schedOpBuffer);
                                              });
    VPUX_THROW_UNLESS(opIdAndSpillWritePair != _opIdAndSpillWritePairs.end(),
                      "No matching spill write operation identified for a given Spill Read (opIdx '{0}')", schedOp.op_);

    auto spillWriteExecOp = opIdAndSpillWritePair->second;

    // Get the insertion point. Pick first non-implicit previous op
    // SpillRead operation will be inserted just after it
    mlir::async::ExecuteOp spillReadInsertionPoint = nullptr;
    auto insertionPointIndex = schedOpIndex;
    while (insertionPointIndex > 0) {
        if (scheduledOps[insertionPointIndex].opType_ == FeasibleMemoryScheduler::EOpType::ORIGINAL_OP) {
            spillReadInsertionPoint = _depsInfo.getExecuteOpAtIndex(scheduledOps[insertionPointIndex].op_);
            break;
        }
        insertionPointIndex--;
    }
    VPUX_THROW_UNLESS(spillReadInsertionPoint != nullptr, "No location to insert Spill Read was identified");

    // In scheduledOpInfo structure op_ identifier for a spillRead operation contains id
    // of the original operation which result had to be spilled
    auto opThatWasSpilled = _depsInfo.getExecuteOpAtIndex(schedOp.op_);
    auto spillReadExecOp = insertSpillReadCopyOp(opThatWasSpilled, spillWriteExecOp, spillReadInsertionPoint,
                                                 schedOp.beginResource(0));

    // After both SpillWrite and SpillRead are inserted update connections
    updateSpillWriteReadUsers(schedOpBuffer, spillWriteExecOp, spillReadExecOp);

    // Remove given spillWrite operation from opId-spillWrite pair vector storage
    // after it was used to prevent from invalid usage once same buffer gets
    // spilled for a second time
    _opIdAndSpillWritePairs.erase(opIdAndSpillWritePair);

    size_t spillReadIndex = _depsInfo.getIndex(spillReadExecOp);
    _log.trace("Spill Read new opId - '{0}'", spillReadIndex);

    // If there are any other spill operations refering to the same op,
    // update them to refer to new spillRead operation
    for (size_t i = schedOpIndex + 1; i < scheduledOps.size(); i++) {
        auto& otherSchedOp = scheduledOps[i];
        if (otherSchedOp.op_ == schedOp.op_ &&
            (otherSchedOp.opType_ == FeasibleMemoryScheduler::EOpType::IMPLICIT_OP_WRITE ||
             otherSchedOp.opType_ == FeasibleMemoryScheduler::EOpType::IMPLICIT_OP_READ) &&
            otherSchedOp.getBuffer(0) == schedOpBuffer) {
            otherSchedOp.op_ = spillReadIndex;
            // Get new output buffer that is the result of spillRead
            otherSchedOp.resourceInfo_[0].buffer_ = getBufferFromAsyncResult(spillReadExecOp.results()[0]);
        }
    }
    // After implicit spillRead operation has been replaced with a proper copy op task then update
    // scheduled ops structure
    schedOp.opType_ = FeasibleMemoryScheduler::EOpType::ORIGINAL_OP;
    schedOp.op_ = spillReadIndex;
    _log = _log.unnest();
}

// This method will go through all scheduled ops and when spill
// operation is identified it will translate it to required CopyOp
void FeasibleMemorySchedulerSpilling::insertSpillCopyOps(
        llvm::SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo>& scheduledOps) {
    _log.trace("Insert Spill CopyOps if needed");
    _log = _log.nest();

    for (size_t i = 0; i < scheduledOps.size(); i++) {
        auto& schedOp = scheduledOps[i];
        if (schedOp.opType_ == FeasibleMemoryScheduler::EOpType::IMPLICIT_OP_WRITE) {
            _log.trace("Spill Write needed for opId - '{0}'", scheduledOps[i].op_);
            createSpillWrite(scheduledOps, i);
        } else if (schedOp.opType_ == FeasibleMemoryScheduler::EOpType::IMPLICIT_OP_READ) {
            _log.trace("Spill Read needed for opId - '{0}'", scheduledOps[i].op_);
            createSpillRead(scheduledOps, i);
        }
    }
    _log = _log.unnest();
    _log.trace("Spill copyOps resolved");
}
