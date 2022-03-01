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

FeasibleMemorySchedulerSpilling::FeasibleMemorySchedulerSpilling(mlir::FuncOp netFunc, IndexedSymbolAttr memSpace,
                                                                 IndexedSymbolAttr secondLvlMemSpace,
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

// Optimize spilling of dataOps. This function will check scheduledOps list and analyze spilling sequence of dataOps.
// Example sequence:
//  1. ORIGINAL (dataOp)
//  ...
//  2. SPILL_WRITE
//  ...
//  3. SPILL_READ
//
// If between ORIGINAL and SPILL_WRITE buffer is not used (e.g. such scenario can happen during prefetching) then
// both 1. and 2. can be removed and SPILL_READ (3.) will be changed to ORIGINAL type.
// TODO:
// If between ORIGINAL and SPILL_WRITE buffer is used only as an input then SPILL_WRITE (2.) can be removed
// and SPILL_READ (3.) turned into equivalent ORIGINAL dataOp
void FeasibleMemorySchedulerSpilling::optimizeDataOpsSpills(
        SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo>& scheduledOps) {
    _log.trace("Optimize data ops spills");
    // Collect information about all data ops that have been spilled
    // For each such dataOp store a sequence of indexes for scheduleOps array
    // where each entry corresponds to related spillWrite/Read operation. First entry
    // (index 0) is the index for original dataOp
    std::unordered_map<FeasibleMemoryScheduler::operationIdxType, SmallVector<size_t>> dataOpSpillTree;
    for (unsigned opIndex = 0; opIndex < scheduledOps.size(); opIndex++) {
        auto& op = scheduledOps[opIndex];
        // Check if this is spillRead/Write of data op
        if ((op.isSpillWrite() || op.isSpillRead()) && op.isDataOp()) {
            // Check if related dataOp has single output. If not
            // then skip this optimization
            if (_depsInfo.getExecuteOpAtIndex(op.op_).results().size() > 1) {
                continue;
            }

            // Find if this is spilling of already encountered dataOp
            auto dataOpIt = dataOpSpillTree.find(op.op_);
            if (dataOpIt != dataOpSpillTree.end()) {
                // If dataOp was already identified store index of related spill operation
                dataOpIt->second.push_back(opIndex);
            } else {
                // If this is spilling of new op, find source op and check if this is dataOp
                int origOpIndex;
                for (origOpIndex = opIndex - 1; origOpIndex >= 0; origOpIndex--) {
                    auto schedOrigOp = scheduledOps[origOpIndex];
                    if (schedOrigOp.isOriginalOp() && schedOrigOp.op_ == op.op_) {
                        // As a first element store index to original operation
                        dataOpSpillTree[schedOrigOp.op_].push_back(origOpIndex);
                        // Store index to identified spillWrite/Read operation
                        dataOpSpillTree[schedOrigOp.op_].push_back(opIndex);
                        break;
                    }
                }
                VPUX_THROW_UNLESS(origOpIndex >= 0,
                                  "Unable to find in scheduled ops original operation for a given spill op '{0}'",
                                  op.op_);
            }
        }
    }

    if (dataOpSpillTree.empty()) {
        _log.trace("No data ops spilling identified");
        return;
    }

    // Dump data ops spilling information
    _log.trace("Data operation spilling sequence:");
    for (auto& dataOp : dataOpSpillTree) {
        _log.nest(1).trace("Operation - '{0}'", dataOp.first);
        for (auto& i : dataOp.second) {
            auto& op = scheduledOps[i];
            _log.nest(2).trace("['{0}']: op = '{1}'\t type = '{2}'\t time = '{3}'", i, op.op_, op.opTypeName(),
                               op.time_);
        }
    }

    SmallVector<size_t> operationIndexesToRemove;
    // Check if between original op / spillRead and spillWrite buffer
    // from dataOp is used by any operation
    _log.trace("Check on possible removal of spills of data operations:");
    for (auto& dataOp : dataOpSpillTree) {
        _log.nest(1).trace("Operation - '{0}'", dataOp.first);
        auto dataOpSpillIndexes = dataOp.second;
        for (size_t i = 0; i < dataOpSpillIndexes.size() - 1; i++) {
            // Check for a sequence origOp/SpillRead -> SpillWrite
            if (scheduledOps[dataOpSpillIndexes[i]].isSpillWrite()) {
                continue;
            }
            if (!scheduledOps[dataOpSpillIndexes[i + 1]].isSpillWrite()) {
                continue;
            }
            auto& origOrSpillReadOpIndex = dataOpSpillIndexes[i];
            auto nextSpillWriteOpIndex = dataOpSpillIndexes[i + 1];

            VPUX_THROW_UNLESS(origOrSpillReadOpIndex < nextSpillWriteOpIndex,
                              "Incorrect order of indexes of spill read and next spill write ops for scheduledOps");

            bool isBufferUsedAsArgument = false;
            bool isBufferUsedAsResult = false;
            auto buffer = scheduledOps[origOrSpillReadOpIndex].getBuffer(0);
            for (size_t schedOpIdx = origOrSpillReadOpIndex + 1; schedOpIdx < nextSpillWriteOpIndex; schedOpIdx++) {
                // TODO: Maybe it would make sense to create a geenric utility function
                // to check if a given buffer is used as a operation input or output
                auto execOp = _depsInfo.getExecuteOpAtIndex(scheduledOps[schedOpIdx].op_);
                // Check if buffer is used for an operation input
                for (auto operand : execOp->getOperands()) {
                    if (operand.getType().isa<mlir::async::ValueType>()) {
                        auto rootBuffers = _aliasInfo.getRoots(operand);
                        VPUX_THROW_UNLESS(rootBuffers.size() == 1,
                                          "Value '{0}' expected to have only one root. Got {1}", operand,
                                          rootBuffers.size());
                        if (*rootBuffers.begin() == buffer) {
                            isBufferUsedAsArgument = true;
                            break;
                        }
                    }
                }
                // Check if buffer is used for an operation output
                for (auto res : execOp.results()) {
                    auto rootBuffers = _aliasInfo.getRoots(res);
                    VPUX_THROW_UNLESS(rootBuffers.size() == 1, "Value '{0}' expected to have only one root. Got {1}",
                                      res, rootBuffers.size());
                    if (*rootBuffers.begin() == buffer) {
                        isBufferUsedAsResult = true;
                        break;
                    }
                }

                if (isBufferUsedAsArgument && isBufferUsedAsResult) {
                    break;
                }
            }

            auto isBufferUsed = isBufferUsedAsArgument || isBufferUsedAsResult;

            // If buffer was not used by any operation in between then given read-write pair is not needed
            // This can happen if scheduler prefetched dataOp which got immediately spilled
            if (!isBufferUsed) {
                _log.nest(2).trace("Buffer not used at all between spillRead/OrigOp '{0}' and next spillWrite op '{1}'",
                                   origOrSpillReadOpIndex, nextSpillWriteOpIndex);
                _log.nest(2).trace("Remove spillRead/OrigOp  - '{0}'", origOrSpillReadOpIndex);
                _log.nest(2).trace("Remove next spillWriteOp - '{0}'", nextSpillWriteOpIndex);

                // Ops can be removed
                operationIndexesToRemove.push_back(origOrSpillReadOpIndex);
                operationIndexesToRemove.push_back(nextSpillWriteOpIndex);

                // If read operation was origOp then change next corresponding operation
                if (scheduledOps[origOrSpillReadOpIndex].isOriginalOp()) {
                    // In such case update next read operation to be original operation
                    for (size_t j = i + 2; j < dataOpSpillIndexes.size(); j++) {
                        auto nextSpillReadIndex = dataOpSpillIndexes[j];
                        if (scheduledOps[nextSpillReadIndex].isSpillRead()) {
                            _log.nest(2).trace("Change next spillRead to origOp - '{0}'", nextSpillReadIndex);
                            scheduledOps[nextSpillReadIndex].opType_ = scheduledOps[origOrSpillReadOpIndex].opType_;
                            break;
                        }
                    }
                }
            } else if (isBufferUsedAsArgument && !isBufferUsedAsResult) {
                // If buffer was used just as an argument then next spillWrite can be removed
                // as buffer state has not changed. Following spillRead needs to be turned
                // to OrigOp
                _log.nest(2).trace("Buffer used as input between spillRead/OrigOp '{0}' and next spillWrite op '{1}'",
                                   origOrSpillReadOpIndex, nextSpillWriteOpIndex);
                // TODO:
                // Besides removing spillWrite need to update next spillRead
                // operationIndexesToRemove.push_back(nextSpillWriteOpIndex);
            }
        }
    }

    // Sort operation indexes
    std::sort(operationIndexesToRemove.begin(), operationIndexesToRemove.end());

    // Remove in reverse order to have indexes valid after erasing entries in scheduledOp
    for (auto opIt = operationIndexesToRemove.rbegin(); opIt != operationIndexesToRemove.rend(); opIt++) {
        scheduledOps.erase(scheduledOps.begin() + *opIt);
    }
    _log.trace("Operations that have been removed - '{0}'", operationIndexesToRemove.size());
}

// This function tries to eliminate redundant spill write operations if exactly the same
// buffer was already spilled before and resides in DDR. In such case subsequent
// spill write can be removed leaving just the needed spill read that will refer
// to first DDR location of spilled buffer
void FeasibleMemorySchedulerSpilling::removeRedundantSpillWrites(
        SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo>& scheduledOps) {
    _log.trace("Remove redundant Spill Writes");

    SmallVector<size_t> spillWriteIndexes;
    SmallVector<size_t> duplicateSpillWriteIndexes;

    // Traverse whole scheduled ops structure and check each spill write/read op
    for (size_t index = 0; index < scheduledOps.size(); index++) {
        auto& op = scheduledOps[index];
        if (op.isSpillWrite()) {
            _log.trace("SPILL WRITE for op '{0}', idx - '{1}'", op.op_, index);
            // For each spill write op check if this is a duplicate by comparing op number and buffer of each
            // previously encountered spill writes
            for (auto spillWriteIndexIt = spillWriteIndexes.rbegin(); spillWriteIndexIt != spillWriteIndexes.rend();
                 spillWriteIndexIt++) {
                if (scheduledOps[*spillWriteIndexIt].op_ == op.op_ &&
                    scheduledOps[*spillWriteIndexIt].getBuffer(0) == op.getBuffer(0)) {
                    // If op and buffer match then duplicate was detected
                    duplicateSpillWriteIndexes.push_back(index);
                    _log.nest().trace(
                            "Duplicate spill for op '{0}': SPILL WRITE idx - '{1}', previous SPILL WRITE idx - '{2}'",
                            op.op_, index, *spillWriteIndexIt);
                    break;
                }
            }
            // Store information about position of each spill write for reference
            spillWriteIndexes.push_back(index);
        }
    }
    _log.trace("Spills detected - '{0}', spill writes to remove - '{1}'", spillWriteIndexes.size(),
               duplicateSpillWriteIndexes.size());

    // Remove in reverse order to have indexes valid after erasing entries in scheduledOp
    for (auto opIt = duplicateSpillWriteIndexes.rbegin(); opIt != duplicateSpillWriteIndexes.rend(); opIt++) {
        scheduledOps.erase(scheduledOps.begin() + *opIt);
    }
}

SmallVector<mlir::Value> FeasibleMemorySchedulerSpilling::getAsyncResultsForBuffer(
        mlir::async::ExecuteOp opThatWasSpilled, mlir::Value buffer) {
    SmallVector<mlir::Value> buffersToCheck = {buffer};
    SmallVector<mlir::Value> asyncResults;

    // Search if this buffer is a replacement for some original buffer which got spilled
    // If such original buffer is located use it for aliases as this information is not updated
    // with new buffers in dependant operations
    for (auto& replacementPairs : _bufferReplacementAfterSpillRead) {
        if (replacementPairs.second == buffer) {
            buffersToCheck.push_back(replacementPairs.first);
        }
    }
    for (auto& bufferToCheck : buffersToCheck) {
        for (auto bufferAlias : _aliasInfo.getAllAliases(bufferToCheck)) {
            if (bufferAlias.getType().isa<mlir::async::ValueType>() &&
                bufferAlias.getDefiningOp() == opThatWasSpilled.getOperation()) {
                asyncResults.push_back(bufferAlias);
            }
        }
    }

    VPUX_THROW_WHEN(asyncResults.empty(),
                    "No async result matched for a given buffer\n buffer - {0}\n op that was spilled - {1}", buffer,
                    opThatWasSpilled);

    return asyncResults;
}

mlir::Value FeasibleMemorySchedulerSpilling::getBufferFromAsyncResult(mlir::Value asyncResult) {
    const auto resultType = asyncResult.getType();
    VPUX_THROW_UNLESS(resultType.isa<mlir::async::ValueType>(), "This is not async result. Got: '{0}'", resultType);
    const auto roots = _aliasInfo.getRoots(asyncResult);
    VPUX_THROW_UNLESS(roots.size() == 1, "Value '{0}' expected to have only one root. Got {1}", asyncResult,
                      roots.size());
    return *roots.begin();
}

mlir::async::ExecuteOp FeasibleMemorySchedulerSpilling::insertSpillWriteCopyOp(mlir::async::ExecuteOp opThatWasSpilled,
                                                                               mlir::async::ExecuteOp insertAfterExecOp,
                                                                               mlir::Value bufferToSpill,
                                                                               size_t allocatedAddress) {
    auto spillWriteNameLoc = appendLoc(opThatWasSpilled->getLoc(),
                                       llvm::formatv("spill_write_{0}", _depsInfo.getIndex(opThatWasSpilled)).str());
    _log.trace("Insert Spill Write copyOp - '{0}'", spillWriteNameLoc);

    // Get spill destination buffer type (memref) from the provided
    // type of source buffer that is to be spilled
    auto getSpillBufferType = [&](vpux::NDTypeInterface type) {
        const auto shape = type.getShape();
        const auto elemType = type.getElementType();
        const auto order = type.getDimsOrder();
        return getMemRefType(shape, elemType, order, _secondLvlMemSpace);
    };
    auto spillBufferType = getSpillBufferType(bufferToSpill.getType());

    // Update address of the buffer that is to be spilled as spillWrite source buffer
    // is not correctly configured during scheduler memory allocation
    _scan.handler().setAddress(bufferToSpill, allocatedAddress);

    // Create buffer in second level memory
    mlir::OpBuilder builder(_allocOpInsertionPoint);
    builder.setInsertionPoint(_allocOpInsertionPoint);
    auto spillBuffer = builder.create<mlir::memref::AllocOp>(spillWriteNameLoc, spillBufferType);

    // Update aliases info for newly created root buffer
    _aliasInfo.addAlias(spillBuffer.memref(), spillBuffer.memref());

    // Create new AsyncExecOp
    builder.setInsertionPointAfter(insertAfterExecOp);
    auto spillWriteExecOp =
            builder.create<mlir::async::ExecuteOp>(spillWriteNameLoc, spillBuffer->getResultTypes(), None, None);

    IERT::CopyOp spillWriteCopyOp;

    auto bodyBlock = &spillWriteExecOp.body().front();
    builder.setInsertionPointToStart(bodyBlock);

    // Build body of spill write async exec op
    if (bufferToSpill.getType().isa<VPUIP::DistributedBufferType>()) {
        // Create NCEClusterTiling with CopyOp in the body of new AsyncExecOp
        SmallVector<mlir::Value> inputsOutputOperands = {bufferToSpill, spillBuffer.memref()};

        const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
            spillWriteCopyOp = builder.create<IERT::CopyOp>(loc, newOperands[0], newOperands[1]);
        };

        auto clusterTilingOp = builder.create<VPUIP::NCEClusterTilingOp>(spillWriteNameLoc, spillBufferType,
                                                                         inputsOutputOperands, bodyBuilder);
        builder.create<mlir::async::YieldOp>(spillWriteNameLoc, clusterTilingOp.results()[0]);
        _aliasInfo.addAlias(spillBuffer.memref(), clusterTilingOp.results()[0]);
    } else {
        // Create CopyOp in the body of new AsyncExecOp
        spillWriteCopyOp = builder.create<IERT::CopyOp>(spillWriteNameLoc, bufferToSpill, spillBuffer.memref());
        builder.create<mlir::async::YieldOp>(spillWriteNameLoc, spillWriteCopyOp->getResults());
    }

    // Update aliases for spillWrite result
    _aliasInfo.addAlias(spillBuffer.memref(), spillWriteCopyOp.output());
    _aliasInfo.addAlias(spillBuffer.memref(), spillWriteExecOp.results()[0]);

    // Update executor attributes of new AsyncExecOp
    auto copyOpExecutor = mlir::dyn_cast_or_null<IERT::AsyncLayerOpInterface>(spillWriteCopyOp.getOperation());
    auto executor = copyOpExecutor.getExecutor();
    if (executor != nullptr) {
        IERT::IERTDialect::setExecutor(spillWriteExecOp, executor);
    }

    // Update dependencies map and get new operation index
    _depsInfo.insertNewExecOpToDepsMap(spillWriteExecOp);

    // Update dependency
    _depsInfo.addDependency(opThatWasSpilled, spillWriteExecOp);

    return spillWriteExecOp;
}

mlir::async::ExecuteOp FeasibleMemorySchedulerSpilling::insertSpillReadCopyOp(mlir::async::ExecuteOp opThatWasSpilled,
                                                                              mlir::Value bufferToSpill,
                                                                              mlir::async::ExecuteOp spillWriteExecOp,
                                                                              mlir::async::ExecuteOp insertAfterExecOp,
                                                                              size_t allocatedAddress) {
    auto spillReadNameLoc = appendLoc(opThatWasSpilled->getLoc(),
                                      llvm::formatv("spill_read_{0}", _depsInfo.getIndex(opThatWasSpilled)).str());
    _log.trace("Insert Spill Read copyOp - '{0}'", spillReadNameLoc);

    // Get information about spill write returned memref type and prepare new one with proper memory location
    auto spillWriteResult = spillWriteExecOp.results()[0];
    auto spillWriteAsyncType = spillWriteResult.getType().dyn_cast<mlir::async::ValueType>();

    // Create buffer in first level memory to bring back spilled buffer
    mlir::OpBuilder builder(_allocOpInsertionPoint);
    builder.setInsertionPoint(_allocOpInsertionPoint);
    mlir::Operation* newBufferOp;
    if (bufferToSpill.getType().isa<VPUIP::DistributedBufferType>()) {
        newBufferOp = builder.create<VPURT::AllocDistributed>(spillReadNameLoc, bufferToSpill.getType());
    } else {
        newBufferOp = builder.create<mlir::memref::AllocOp>(spillReadNameLoc,
                                                            bufferToSpill.getType().cast<mlir::MemRefType>());
    }
    auto newBufferResult = newBufferOp->getResult(0);

    // Update aliases info for newly created root buffer
    _aliasInfo.addAlias(newBufferResult, newBufferResult);

    // Configure address as prepared by scheduler.
    // Since it is a new buffer it was not assigned before
    _scan.handler().setAddress(newBufferResult, allocatedAddress);

    // Store information about what buffer replaces original buffer that was marked for spilling
    // If such replacement pair already exists, then update it with a new buffer
    bool replacementPairFound = false;
    for (auto& replacementPairs : _bufferReplacementAfterSpillRead) {
        if (replacementPairs.second == bufferToSpill) {
            replacementPairs.second = newBufferResult;
            replacementPairFound = true;
            break;
        }
    }
    // If this buffer didn't correspond to any exisitng buffer replacement pair then insert a new one
    if (!replacementPairFound) {
        _bufferReplacementAfterSpillRead.insert({bufferToSpill, newBufferResult});
    }

    // Create new AsyncExecOp in correct place
    builder.setInsertionPointAfter(insertAfterExecOp);
    auto spillReadExecOp =
            builder.create<mlir::async::ExecuteOp>(spillReadNameLoc, newBufferOp->getResultTypes(), None, None);

    // Update operands of new AsyncExecOp to contain result of AsyncExecOp of spillWrite
    spillReadExecOp.operandsMutable().append(spillWriteResult);
    auto innerAsyncArgForSpill = spillReadExecOp.getBody()->addArgument(spillWriteAsyncType.getValueType());

    IERT::CopyOp spillReadCopyOp;

    auto bodyBlock = &spillReadExecOp.body().front();
    builder.setInsertionPointToStart(bodyBlock);

    // Build body of spill read async exec op
    if (bufferToSpill.getType().isa<VPUIP::DistributedBufferType>()) {
        // Create NCEClusterTiling with CopyOp in the body of new AsyncExecOp
        SmallVector<mlir::Value> inputsOutputOperands = {innerAsyncArgForSpill, newBufferResult};

        const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
            spillReadCopyOp = builder.create<IERT::CopyOp>(loc, newOperands[0], newOperands[1]);
        };

        auto clusterTilingOp = builder.create<VPUIP::NCEClusterTilingOp>(spillReadNameLoc, bufferToSpill.getType(),
                                                                         inputsOutputOperands, bodyBuilder);
        builder.create<mlir::async::YieldOp>(spillReadNameLoc, clusterTilingOp.results()[0]);
        _aliasInfo.addAlias(newBufferResult, clusterTilingOp.results()[0]);
    } else {
        // Create CopyOp in the body of new AsyncExecOp
        spillReadCopyOp = builder.create<IERT::CopyOp>(spillReadNameLoc, innerAsyncArgForSpill, newBufferResult);
        builder.create<mlir::async::YieldOp>(spillReadNameLoc, spillReadCopyOp->getResults());
    }

    // Update alias for spillRead result
    _aliasInfo.addAlias(newBufferResult, spillReadCopyOp.output());
    _aliasInfo.addAlias(newBufferResult, spillReadExecOp.results()[0]);

    // Update executor attributes of new AsyncExecOp
    auto copyOpExecutor = mlir::dyn_cast_or_null<IERT::AsyncLayerOpInterface>(spillReadCopyOp.getOperation());
    auto executor = copyOpExecutor.getExecutor();
    if (executor != nullptr) {
        IERT::IERTDialect::setExecutor(spillReadExecOp, executor);
    }

    // Update dependencies map and get new operation index
    _depsInfo.insertNewExecOpToDepsMap(spillReadExecOp);

    // Update dependency
    _depsInfo.addDependency(spillWriteExecOp, spillReadExecOp);

    return spillReadExecOp;
}

mlir::Operation* FeasibleMemorySchedulerSpilling::SpillUsersUpdate::getViewOpForMasterBuffer(mlir::Value asyncResult) {
    // Identify pure viewOp for master buffer that is related to result of
    // asyncExecOp that is spilled
    mlir::Operation* viewOpForMasterBuffer = nullptr;

    mlir::Value sourceAlias = asyncResult;
    while ((sourceAlias = _spillingParentClass._aliasInfo.getSource(sourceAlias))) {
        auto sourceOp = sourceAlias.getDefiningOp();
        if (IERT::isPureViewOp(sourceOp)) {
            if (auto viewOp = mlir::dyn_cast<mlir::ViewLikeOpInterface>(sourceOp)) {
                if (viewOp.getViewSource() == _bufferToSpill) {
                    VPUX_THROW_UNLESS(
                            viewOpForMasterBuffer == nullptr,
                            "Chain of pure-view ops is not supported for the same buffer in single async.ExecuteOp. "
                            "Pure-view has already been identified for given asyncResult - {0}",
                            asyncResult);
                    viewOpForMasterBuffer = viewOp.getOperation();
                }
            }
        }
    }

    return viewOpForMasterBuffer;
}

SmallVector<mlir::async::ExecuteOp>
FeasibleMemorySchedulerSpilling::SpillUsersUpdate::getUsersOfSpilledOpThatNeedUpdate(
        mlir::Value opThatWasSpilledResult) {
    // Get all asyncExecOps that are users of result of spilled op and appear in
    // IR after spillRead. Those users would need to be updated to refer to result
    // of spillRead
    SmallVector<mlir::async::ExecuteOp> usersOfSpilledOpThatNeedUpdate;
    for (auto* user : opThatWasSpilledResult.getUsers()) {
        if (mlir::isa_and_nonnull<mlir::async::ExecuteOp>(user) && !user->isBeforeInBlock(_spillReadExecOp)) {
            usersOfSpilledOpThatNeedUpdate.push_back(mlir::dyn_cast_or_null<mlir::async::ExecuteOp>(user));
        }
    }
    return usersOfSpilledOpThatNeedUpdate;
}

unsigned int FeasibleMemorySchedulerSpilling::SpillUsersUpdate::getOperandIndexForSpillResultUser(
        mlir::async::ExecuteOp spillResultUser, mlir::Value spilledAsyncResult) {
    // For a given user of result of spilled operation identify the
    // operand index for this dependency
    unsigned int operandIndex = 0;
    bool operandFound = false;
    for (const auto& operand : spillResultUser.getOperands()) {
        if (operand.getType().isa<mlir::async::ValueType>()) {
            if (operand == spilledAsyncResult) {
                operandFound = true;
                break;
            }
            operandIndex++;
        }
    }
    VPUX_THROW_UNLESS(operandFound, "Unable to find async.ExecOp operand index matching result of op that was spilled");
    return operandIndex;
}

void FeasibleMemorySchedulerSpilling::SpillUsersUpdate::updateSpillResultUsers(mlir::Value oldResult,
                                                                               mlir::Value newResult) {
    // Find operations which should be excluded from operand update to result of spillRead.
    // Those are all operations which appear in IR before spillRead
    llvm::SmallPtrSet<mlir::Operation*, 1> excludedUsersFromOperandsUpdate;
    for (auto* user : oldResult.getUsers()) {
        if (mlir::isa_and_nonnull<mlir::async::ExecuteOp>(user) &&
            user->isBeforeInBlock(_spillReadExecOp.getOperation())) {
            excludedUsersFromOperandsUpdate.insert(user);
        }
    }

    // Update connections opThatWasSpilled -> SpillWrite -> SpillRead -> UserOfSpilledBuffer
    oldResult.replaceAllUsesExcept(newResult, excludedUsersFromOperandsUpdate);
}

void FeasibleMemorySchedulerSpilling::SpillUsersUpdate::updateSpillBufferUsers(mlir::Value oldBuffer,
                                                                               mlir::Value newBuffer) {
    // Get information about the users of original output buffer that should still refer to it
    // (e.g. operations that appear in IR before)
    llvm::SmallPtrSet<mlir::Operation*, 1> excludedUsersFromOrigBufferUpdate;
    for (auto* user : oldBuffer.getUsers()) {
        if (user != nullptr) {
            if (user->getParentOp()->isBeforeInBlock(_spillReadExecOp)) {
                excludedUsersFromOrigBufferUpdate.insert(user);
            }
        }
    }

    // Update all users of original output buffer with the new buffer from spillRead except
    // the operations which were identified to refer to old output buffer
    oldBuffer.replaceAllUsesExcept(newBuffer, excludedUsersFromOrigBufferUpdate);
}

void FeasibleMemorySchedulerSpilling::SpillUsersUpdate::resolveSpillBufferUsage() {
    auto opThatWasSpilledResults = _spillingParentClass.getAsyncResultsForBuffer(_opThatWasSpilled, _bufferToSpill);

    auto spillReadExecOpResult = _spillReadExecOp.results()[0];

    // Users referring to spilled buffer need to be properly updated to now refer to result of spillRead
    for (auto& opThatWasSpilledResult : opThatWasSpilledResults) {
        auto usersOfSpilledOpThatNeedUpdate = getUsersOfSpilledOpThatNeedUpdate(opThatWasSpilledResult);

        // Identify pure viewOp for master buffer that is related to result of asyncExecOp
        // that is spilled. If such operation is located then users need to have
        // similar operation injected to properly refer to replacement of spilled buffer
        auto* viewOpForMasterBuffer = getViewOpForMasterBuffer(opThatWasSpilledResult);
        if (viewOpForMasterBuffer && !usersOfSpilledOpThatNeedUpdate.empty()) {
            for (auto userOfSpilledOpThatNeedUpdate : usersOfSpilledOpThatNeedUpdate) {
                auto userOfSpilledOpBodyBlock = &userOfSpilledOpThatNeedUpdate.body().front();
                // Insert view Op defining relation between spilled buffer and user of
                // asyncExecOp result referring to this buffer
                mlir::OpBuilder builder(userOfSpilledOpThatNeedUpdate);
                builder.setInsertionPointToStart(userOfSpilledOpBodyBlock);
                auto newViewOp = builder.clone(*viewOpForMasterBuffer);

                // Get asyncExecOp argument index related to result of spilled asyncExecOp
                auto operandIndex =
                        getOperandIndexForSpillResultUser(userOfSpilledOpThatNeedUpdate, opThatWasSpilledResult);

                // Get argument of asyncExecOp block that would need to be updated
                // to be used in the body through newly inserted view op
                auto arg = userOfSpilledOpBodyBlock->getArgument(operandIndex);
                arg.replaceAllUsesWith(newViewOp->getOpResult(0));

                auto finalType = newViewOp->getOpOperand(0).get().getType();
                newViewOp->setOperand(0, arg);
                newViewOp->getOpOperand(0).get().setType(finalType);
            }
        }

        updateSpillResultUsers(opThatWasSpilledResult, spillReadExecOpResult);
    }

    // Get new output buffer that is the result of spillRead
    auto newOutputBuffer = _spillingParentClass.getBufferFromAsyncResult(spillReadExecOpResult);

    // If there are operations which were referring directly to output buffer that was spilled
    // they should be updated to refer to result of spillRead if they appear in the IR
    // after the op whose result was spilled
    updateSpillBufferUsers(_bufferToSpill, newOutputBuffer);
}

// This function will update operands of users of spilled buffer
// and make proper connections
void FeasibleMemorySchedulerSpilling::updateSpillWriteReadUsers(mlir::Value bufferToSpill,
                                                                mlir::async::ExecuteOp spillWriteExecOp,
                                                                mlir::async::ExecuteOp spillReadExecOp) {
    _log.trace("Update users of Spill Write-Read pair: '{0}' -> '{1}'", spillWriteExecOp->getLoc(),
               spillReadExecOp->getLoc());

    // Find asyncExecOps which have result corresponding to buffer that got spilled
    SmallVector<mlir::async::ExecuteOp> opsThatWereSpilled;
    for (auto bufferAlias : _aliasInfo.getAllAliases(bufferToSpill)) {
        if (bufferAlias.getType().isa<mlir::async::ValueType>()) {
            if (const auto execOpWithSpilledResult =
                        mlir::dyn_cast<mlir::async::ExecuteOp>(bufferAlias.getDefiningOp())) {
                if (execOpWithSpilledResult->isBeforeInBlock(spillReadExecOp)) {
                    opsThatWereSpilled.push_back(execOpWithSpilledResult);
                }
            }
        }
    }

    std::sort(opsThatWereSpilled.begin(), opsThatWereSpilled.end(),
              [](mlir::async::ExecuteOp execOp1, mlir::async::ExecuteOp execOp2) {
                  return execOp2.getOperation()->isBeforeInBlock(execOp1.getOperation());
              });

    for (auto& opThatWasSpilled : opsThatWereSpilled) {
        _log.trace("Resolve users of operation: '{0}'", opThatWasSpilled->getLoc());
        SpillUsersUpdate spillUsersUpdateHandler(*this, opThatWasSpilled, spillReadExecOp, bufferToSpill);
        spillUsersUpdateHandler.resolveSpillBufferUsage();
    }
}

// Create Spill Write operation based on data from feasible scheduler
void FeasibleMemorySchedulerSpilling::createSpillWrite(
        SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo>& scheduledOps, size_t schedOpIndex) {
    auto& schedOp = scheduledOps[schedOpIndex];
    auto schedOpBuffer = schedOp.getBuffer(0);
    _log = _log.nest();
    _log.trace("Create Spill Write for buffer - '{0}'", schedOpBuffer);

    // Get the insertion point. Pick first non-implicit previous op
    // SpillWrite operation will be inserted just after it
    mlir::async::ExecuteOp spillWriteInsertionPoint = nullptr;
    auto insertionPointIndex = schedOpIndex;
    while (insertionPointIndex > 0) {
        if (scheduledOps[insertionPointIndex].isOriginalOp()) {
            spillWriteInsertionPoint = _depsInfo.getExecuteOpAtIndex(scheduledOps[insertionPointIndex].op_);
            break;
        }
        insertionPointIndex--;
    }
    VPUX_THROW_UNLESS(spillWriteInsertionPoint != nullptr, "No location to insert Spill Write was identified");

    // In scheduledOpInfo structure op_ identifier for a spillWrite operation contains id
    // of the original operation which result had to be spilled
    auto opThatWasSpilled = _depsInfo.getExecuteOpAtIndex(schedOp.op_);

    auto spillBuffer = schedOpBuffer;
    if (_bufferReplacementAfterSpillRead.find(schedOpBuffer) != _bufferReplacementAfterSpillRead.end()) {
        spillBuffer = _bufferReplacementAfterSpillRead[schedOpBuffer];
        _log.trace("Actual buffer for Spill Write - '{0}'", spillBuffer);
    }

    auto spillWriteExecOp =
            insertSpillWriteCopyOp(opThatWasSpilled, spillWriteInsertionPoint, spillBuffer, schedOp.beginResource(0));
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
        SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo>& scheduledOps, size_t schedOpIndex) {
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
        if (scheduledOps[insertionPointIndex].isOriginalOp()) {
            spillReadInsertionPoint = _depsInfo.getExecuteOpAtIndex(scheduledOps[insertionPointIndex].op_);
            break;
        }
        insertionPointIndex--;
    }
    VPUX_THROW_UNLESS(spillReadInsertionPoint != nullptr, "No location to insert Spill Read was identified");

    // In scheduledOpInfo structure op_ identifier for a spillRead operation contains id
    // of the original operation which result had to be spilled
    auto opThatWasSpilled = _depsInfo.getExecuteOpAtIndex(schedOp.op_);

    auto spillBuffer = schedOpBuffer;
    if (_bufferReplacementAfterSpillRead.find(schedOpBuffer) != _bufferReplacementAfterSpillRead.end()) {
        spillBuffer = _bufferReplacementAfterSpillRead[schedOpBuffer];
        _log.trace("Actual buffer for Spill Read - '{0}'", spillBuffer);
    }
    auto spillReadExecOp = insertSpillReadCopyOp(opThatWasSpilled, spillBuffer, spillWriteExecOp,
                                                 spillReadInsertionPoint, schedOp.beginResource(0));

    // After both SpillWrite and SpillRead are inserted update connections
    updateSpillWriteReadUsers(spillBuffer, spillWriteExecOp, spillReadExecOp);

    size_t spillReadIndex = _depsInfo.getIndex(spillReadExecOp);
    _log.trace("Spill Read new opId - '{0}'", spillReadIndex);

    // If there are any other spill operations refering to the same op,
    // update them to refer to new spillRead operation
    for (size_t i = schedOpIndex + 1; i < scheduledOps.size(); i++) {
        auto& otherSchedOp = scheduledOps[i];
        if (otherSchedOp.op_ == schedOp.op_ && (otherSchedOp.isSpillWrite() || otherSchedOp.isSpillRead()) &&
            otherSchedOp.getBuffer(0) == schedOpBuffer) {
            otherSchedOp.op_ = spillReadIndex;
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
        SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo>& scheduledOps) {
    _log.trace("Insert Spill CopyOps if needed");
    _log = _log.nest();

    for (size_t i = 0; i < scheduledOps.size(); i++) {
        auto& schedOp = scheduledOps[i];
        if (schedOp.isSpillWrite()) {
            _log.trace("Spill Write needed for opId - '{0}'", scheduledOps[i].op_);
            createSpillWrite(scheduledOps, i);
        } else if (schedOp.isSpillRead()) {
            _log.trace("Spill Read needed for opId - '{0}'", scheduledOps[i].op_);
            createSpillRead(scheduledOps, i);
        }
    }
    _log = _log.unnest();
    _log.trace("Spill copyOps resolved");
}
