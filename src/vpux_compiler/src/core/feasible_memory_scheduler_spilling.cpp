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

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include "vpux/utils/core/range.hpp"

using namespace vpux;

//
// Feasible Memory Scheduler Spilling support
//

FeasibleMemorySchedulerSpilling::FeasibleMemorySchedulerSpilling(mlir::FuncOp& netFunc, mlir::Attribute& memSpace,
                                                                 mlir::Attribute& secondLvlMemSpace,
                                                                 AsyncDepsInfo& depsInfo,
                                                                 LinearScan<mlir::Value, LinearScanHandler>& scan,
                                                                 Logger& log)
        : _log(log), _memSpace(memSpace), _secondLvlMemSpace(secondLvlMemSpace), _depsInfo(depsInfo), _scan(scan) {
    // Locate first async-exec-op that will be used to determine insertion point for
    // new allocation operations
    _allocOpInsertionPoint = *(netFunc.getOps<mlir::async::ExecuteOp>().begin());
}

// TODO: Maybe locate such function in a different place
static llvm::SmallVector<std::pair<mlir::Operation*, mlir::Value>> getInnerOpAndOutputBuffersOfMemType(
        mlir::async::ExecuteOp execOp, mlir::Attribute memSpace) {
    llvm::SmallVector<std::pair<mlir::Operation*, mlir::Value>> opAndOutputBufferPairs;

    auto* bodyBlock = &execOp.body().front();
    for (auto& op : bodyBlock->getOperations()) {
        if (mlir::isa<mlir::ViewLikeOpInterface>(op) && mlir::isa<IERT::LayerOpInterface>(op)) {
            auto outputs = mlir::dyn_cast<IERT::LayerOpInterface>(op).getOutputs();
            for (const auto& output : outputs) {
                const auto type = output.getType().dyn_cast<mlir::MemRefType>();
                if (type == nullptr || type.getMemorySpace() != memSpace) {
                    continue;
                }
                std::pair<mlir::Operation*, mlir::Value> opAndOutputBufferPair;
                opAndOutputBufferPair.first = &op;
                opAndOutputBufferPair.second = output;
                opAndOutputBufferPairs.push_back(opAndOutputBufferPair);
            }
        }
    }

    VPUX_THROW_UNLESS(opAndOutputBufferPairs.size() == 1,
                      "Spilling is not supported in case operation has multiple outputs");

    return opAndOutputBufferPairs;
}

mlir::async::ExecuteOp FeasibleMemorySchedulerSpilling::insertSpillWriteCopyOp(mlir::async::ExecuteOp opThatWasSpilled,
                                                                               mlir::async::ExecuteOp insertAfterExecOp,
                                                                               size_t allocatedAddress) {
    std::cout << "Mateusz: insertSpillWriteCopyOp start\n";

    auto spillWriteName = llvm::formatv("spill_write").str();
    auto spillWriteNameLoc = appendLoc(opThatWasSpilled->getLoc(), spillWriteName);
    std::cout << " Mateusz: name - " << stringifyLocation(spillWriteNameLoc) << "\n";
    std::cout << " Mateusz: src_addr - " << allocatedAddress << "\n";

    // Get information about current returned memref type and prepare new one with proper memory location
    auto opToSpillResult = opThatWasSpilled->getResult(1);
    auto opToSpillAsyncType = opToSpillResult.getType().dyn_cast<mlir::async::ValueType>();
    auto opToSpillMemRefType = opToSpillAsyncType.getValueType().cast<mlir::MemRefType>();
    auto spillBufferMemType = changeMemSpace(opToSpillMemRefType, _secondLvlMemSpace);

    // Update address of the buffer that is to be spilled as spillWrite source buffer
    // is not correctly configured during scheduler memory allocation
    _scan.handler().setAddress(getInnerOpAndOutputBuffersOfMemType(opThatWasSpilled, _memSpace)[0].second,
                               allocatedAddress);

    // Create buffer in second level memory
    mlir::OpBuilder builder(_allocOpInsertionPoint);
    builder.setInsertionPointAfter(_allocOpInsertionPoint);
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

    // Update executor attributes of new AsyncExecOp
    uint32_t numExecutorUnits = 0;
    auto copyOpExecutor = mlir::dyn_cast_or_null<IERT::AsyncLayerOpInterface>(spillWriteCopyOp.getOperation());
    auto executor = copyOpExecutor.getExecutor(numExecutorUnits);
    if (executor != nullptr) {
        IERT::IERTDialect::setExecutor(spillWriteExecOp, executor, numExecutorUnits);
    }

    // Update dependencies map and get new operation index
    auto spillWriteExecOpIndex = _depsInfo.insertNewExecOpToDepsMap(spillWriteExecOp);

    std::cout << " Created new exec op index - " << spillWriteExecOpIndex << "\n";

    std::cout << " Mateusz: op chain:\n";
    std::cout << " ------------------------------------ \n";
    opThatWasSpilled.dump();
    std::cout << " ------------------------------------ \n";
    spillBuffer.dump();
    std::cout << " ------------------------------------ \n";
    spillWriteExecOp.dump();
    std::cout << " ------------------------------------ \n";

    std::cout << "Mateusz: insertSpillWriteCopyOp end\n";
    return spillWriteExecOp;
}

mlir::async::ExecuteOp FeasibleMemorySchedulerSpilling::insertSpillReadCopyOp(mlir::async::ExecuteOp opThatWasSpilled,
                                                                              mlir::async::ExecuteOp spillWriteExecOp,
                                                                              mlir::async::ExecuteOp insertBeforeExecOp,
                                                                              size_t allocatedAddress) {
    std::cout << "Mateusz: insertSpillReadCopyOp start\n";

    auto spillReadName = llvm::formatv("spill_read").str();
    auto spillReadNameLoc = appendLoc(opThatWasSpilled->getLoc(), spillReadName);
    std::cout << " Mateusz: name - " << stringifyLocation(spillReadNameLoc) << "\n";
    std::cout << " Mateusz: dst_addr - " << allocatedAddress << "\n";

    // Get information about spill write returned memref type and prepare new one with proper memory location
    auto spillWriteResult = spillWriteExecOp->getResult(1);
    auto spillWriteAsyncType = spillWriteResult.getType().dyn_cast<mlir::async::ValueType>();
    auto spillWriteMemRefType = spillWriteAsyncType.getValueType().cast<mlir::MemRefType>();
    auto newBufferMemType = changeMemSpace(spillWriteMemRefType, _memSpace);

    // Create buffer in first level memory to bring back spilled buffer. Configure its
    // address as since it is a new buffer corresponding AllocOp operation was not set
    // an address
    // mlir::OpBuilder builder(insertBeforeExecOp);
    // builder.setInsertionPoint(insertBeforeExecOp);
    mlir::OpBuilder builder(_allocOpInsertionPoint);
    builder.setInsertionPointAfter(_allocOpInsertionPoint);

    // mateusz
    // auto newBuffer = builder.create<IERT::StaticAllocOp>(spillReadNameLoc, newBufferMemType, allocatedAddress);
    auto newBuffer = builder.create<mlir::memref::AllocOp>(spillReadNameLoc, newBufferMemType);
    _scan.handler().setAddress(newBuffer.memref(), allocatedAddress);

    // Create new AsyncExecOp in correct place
    builder.setInsertionPoint(insertBeforeExecOp);
    // builder.setInsertionPointAfter(newBuffer);
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

    // Update executor attributes of new AsyncExecOp
    uint32_t numExecutorUnits = 0;
    auto copyOpExecutor = mlir::dyn_cast_or_null<IERT::AsyncLayerOpInterface>(spillReadCopyOp.getOperation());
    auto executor = copyOpExecutor.getExecutor(numExecutorUnits);
    if (executor != nullptr) {
        IERT::IERTDialect::setExecutor(spillReadExecOp, executor, numExecutorUnits);
    }

    // Update dependencies map and get new operation index
    auto spillReadExecOpIndex = _depsInfo.insertNewExecOpToDepsMap(spillReadExecOp);

    std::cout << " Created new exec op index - " << spillReadExecOpIndex << "\n";

    std::cout << " Mateusz: op chain:\n";
    std::cout << " ------------------------------------ \n";
    spillWriteExecOp.dump();
    std::cout << " ------------------------------------ \n";
    newBuffer.dump();
    std::cout << " ------------------------------------ \n";
    spillReadExecOp.dump();
    std::cout << " ------------------------------------ \n";

    std::cout << "Mateusz: insertSpillReadCopyOp end\n";
    return spillReadExecOp;
}

// This function will update operands of users of spilled buffer
// and mak proper connections
void FeasibleMemorySchedulerSpilling::updateSpillWriteReadUsers(mlir::async::ExecuteOp opThatWasSpilled,
                                                                mlir::async::ExecuteOp spillWriteExecOp,
                                                                mlir::async::ExecuteOp spillReadExecOp) {
    // Find operations which should be exluded from operand update to result of spillRead.
    // By default this is always spillWrite operation
    llvm::SmallPtrSet<mlir::Operation*, 1> excludedUsersFromOperandsUpdate = {spillWriteExecOp.getOperation()};
    for (auto* user : opThatWasSpilled->getResult(1).getUsers()) {
        if (user != nullptr && user->isBeforeInBlock(spillWriteExecOp.getOperation())) {
            excludedUsersFromOperandsUpdate.insert(user);
        }
    }

    // Update connections opThatWasSpilled -> SpillWrite -> SpillRead -> UserOfSpilledBuffer
    opThatWasSpilled->getResult(1).replaceAllUsesExcept(spillReadExecOp->getResult(1), excludedUsersFromOperandsUpdate);

    // Add tokens matching those new data dependencies
    for (auto* user : spillReadExecOp->getResult(1).getUsers()) {
        if (auto userAsyncOp = mlir::dyn_cast_or_null<mlir::async::ExecuteOp>(user)) {
            userAsyncOp.dependenciesMutable().append(makeArrayRef(spillReadExecOp.token()));
        }
    }

    // If there are operations which were refering directly to output buffer that was spilled
    // they shoud dbe updated to refer to result of spillRead if they appear in the IR
    // after the op whose result was spilled
    // Get information about the users of original output buffer that should still refer to it
    // (e.g. operations that appear in IR before)
    llvm::SmallPtrSet<mlir::Operation*, 1> excludedUsersFromOrigBufferUpdate;
    auto origOpAndOutputBuffer = getInnerOpAndOutputBuffersOfMemType(opThatWasSpilled, _memSpace)[0];
    for (auto* user : origOpAndOutputBuffer.second.getUsers()) {
        if (user != nullptr) {
            std::cout << "  Mateusz: user - " << user->getName().getStringRef().data() << "\n";
            if (user == origOpAndOutputBuffer.first) {
                std::cout << "    Mateusz: orig user\n";
                excludedUsersFromOrigBufferUpdate.insert(user);
            } else if (user->isBeforeInBlock(origOpAndOutputBuffer.first)) {
                std::cout << "    Mateusz: user before orig op\n";
                excludedUsersFromOrigBufferUpdate.insert(user);
            }
        }
    }

    // Get new ouptut buffer that is the result of spillRead
    mlir::Value newOutputBuffer = getInnerOpAndOutputBuffersOfMemType(spillReadExecOp, _memSpace)[0].second;

    // Update all users of original output buffer with the new buffer from spillRead except
    // the operations which were identified to refer to old outptu buffer
    origOpAndOutputBuffer.second.replaceAllUsesExcept(newOutputBuffer, excludedUsersFromOrigBufferUpdate);
}

// This method will go through all scheduled ops and when spill
// operation is identified it will translate it to required CopyOp
void FeasibleMemorySchedulerSpilling::insertSpillCopyOps(
        llvm::SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo>& scheduledOps) {
    std::cout << "Mateusz: insertSpillCopyOps start\n";
    size_t spillCounter = 0;

    for (size_t i = 0; i < scheduledOps.size(); i++) {
        auto& schedOp = scheduledOps[i];
        if (schedOp.opType_ == FeasibleMemoryScheduler::EOpType::IMPLICIT_OP_WRITE) {
            // Get the insertion point. Pick first non-implicit previous op
            // with time t - 1. SpillWrite operation will be inserted just after it
            mlir::async::ExecuteOp spillWriteInsertionPoint = nullptr;
            auto insertionPointIndex = i;
            while (insertionPointIndex > 0) {
                if (scheduledOps[insertionPointIndex].time_ == schedOp.time_ - 1 &&
                    scheduledOps[insertionPointIndex].opType_ == FeasibleMemoryScheduler::EOpType::ORIGINAL_OP) {
                    spillWriteInsertionPoint = _depsInfo.getExecuteOpAtIndex(scheduledOps[insertionPointIndex].op_);
                    break;
                }
                insertionPointIndex--;
            }
            VPUX_THROW_UNLESS(spillWriteInsertionPoint != nullptr, "No location to insert spill write was identified");

            // In scheduledOpInfo structure op_ identifier for a spillWrite operation contains id
            // of the original operation which result had to be spilled
            auto opThatWasSpilled = _depsInfo.getExecuteOpAtIndex(schedOp.op_);
            std::cout << "scheduler address " << schedOp.beginResource() << std::endl;
            auto spillWriteExecOp =
                    insertSpillWriteCopyOp(opThatWasSpilled, spillWriteInsertionPoint, schedOp.beginResource());
            _opIdAndSpillWritePairs.push_back({schedOp.op_, spillWriteExecOp});

            // After implicit spill write operation has been replaced with a proper copy op task then update
            // scheduled ops structure
            schedOp.opType_ = FeasibleMemoryScheduler::EOpType::ORIGINAL_OP;
            schedOp.op_ = _depsInfo.getIndex(spillWriteExecOp);
            schedOp.resourceInfo_.invalidate();
        } else if (schedOp.opType_ == FeasibleMemoryScheduler::EOpType::IMPLICIT_OP_READ) {
            // Get spillWrite operation for the given spillRead to properly
            // connect both operations
            auto opIdAndSpillWritePair = std::find_if(_opIdAndSpillWritePairs.begin(), _opIdAndSpillWritePairs.end(),
                                                      [&](std::pair<size_t, mlir::async::ExecuteOp> pairElemenet) {
                                                          return (pairElemenet.first == schedOp.op_);
                                                      });
            VPUX_THROW_UNLESS(opIdAndSpillWritePair != _opIdAndSpillWritePairs.end(),
                              "No matching spill write operation identifed for a given spill read");

            auto spillWriteExecOp = opIdAndSpillWritePair->second;

            // Get spillRead insertion location which will be the first next
            // non implicit operation with start time t+1. SpillRead task
            // will be inserted just before this operation
            mlir::async::ExecuteOp spillReadInsertionPoint = nullptr;
            auto insertionPointIndex = i;
            while (insertionPointIndex < scheduledOps.size()) {
                if (scheduledOps[insertionPointIndex].time_ == schedOp.time_ + 1 &&
                    scheduledOps[insertionPointIndex].opType_ == FeasibleMemoryScheduler::EOpType::ORIGINAL_OP) {
                    spillReadInsertionPoint = _depsInfo.getExecuteOpAtIndex(scheduledOps[insertionPointIndex].op_);
                    break;
                }
                insertionPointIndex++;
            }
            VPUX_THROW_UNLESS(spillReadInsertionPoint != nullptr, "No location to insert spill read was identified");

            // In scheduledOpInfo structure op_ identifier for a spillRead operation contains id
            // of the original operation which result had to be spilled
            auto opThatWasSpilled = _depsInfo.getExecuteOpAtIndex(schedOp.op_);
            auto spillReadExecOp = insertSpillReadCopyOp(opThatWasSpilled, spillWriteExecOp, spillReadInsertionPoint,
                                                         schedOp.beginResource());

            // After both SpillWrite and SpillRead are inserted update connections
            updateSpillWriteReadUsers(opThatWasSpilled, spillWriteExecOp, spillReadExecOp);

            // Remove given spillWrite operation from opId-spillWrite pair vector storage
            // after it was used to prevent from invalid usage once same buffer gets
            // spilled for a second time
            _opIdAndSpillWritePairs.erase(opIdAndSpillWritePair);

            size_t spillReadIndex = _depsInfo.getIndex(spillReadExecOp);
            // If there are any other spill operations refering to the same op,
            // update them to refer to new spillRead operation
            for (size_t j = i + 1; j < scheduledOps.size(); j++) {
                auto& otherSchedOp = scheduledOps[j];
                if (otherSchedOp.op_ == schedOp.op_ &&
                    (otherSchedOp.opType_ == FeasibleMemoryScheduler::EOpType::IMPLICIT_OP_WRITE ||
                     otherSchedOp.opType_ == FeasibleMemoryScheduler::EOpType::IMPLICIT_OP_READ)) {
                    std::cout << "  Mateusz: Update index of other spill op from " << otherSchedOp.op_ << " to "
                              << spillReadIndex << "\n";
                    otherSchedOp.op_ = spillReadIndex;
                }
            }
            // After implicit spillRead operation has been replaced with a proper copy op task then update
            // scheduled ops structure
            schedOp.opType_ = FeasibleMemoryScheduler::EOpType::ORIGINAL_OP;
            schedOp.op_ = spillReadIndex;
            spillCounter++;
        }
    }
    std::cout << "Mateusz: spillCounter = " << spillCounter << "\n";
    std::cout << "Mateusz: insertSpillCopyOps end\n";
}