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

#include "vpux/compiler/core/feasible_memory_scheduler.hpp"

#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include "vpux/utils/core/range.hpp"

using namespace vpux;
using operationIdxType = FeasibleMemoryScheduler::operationIdxType;

//
// Feasible Memory Scheduler
//

// This class will try to produce a feasible memory schedule based on the dependency map provided from
// AsyncDepsInfo and use the LinearScan class to allocate the resources.
// Data and Compute ops, where Data ops are operations moving data to CMX are distinguished in order to
// follow the scheduling of Compute ops along with their dependencies (Data ops). This optimizes CMX usage,
// and allows for feasible CMX schedule to be generated.
// The graph is iterated topologically based on the dependencies from input to output(s).
// In init() the input will be considered as a compute operation, this will be the first ready compute operation.
// In nextSchedulableOp() there are two possible scenarios:
// 1. Scheduling the next earliest operation from the start time heap, and adding it to the op output table.
// 2. Unscheduling operations: freeing CMX space and updating dependencies, creating new ready
//      operations which will be allocated at the next time slot.

FeasibleMemoryScheduler::FeasibleMemoryScheduler(mlir::Attribute& memSpace, MemLiveRangeInfo& liveRangeInfo,
                                                 AsyncDepsInfo& depsInfo, AliasesInfo& aliasInfo, Logger log,
                                                 LinearScan<mlir::Value, LinearScanHandler>& scan)
        : _log(log),
          _memSpace(memSpace),
          _liveRangeInfo(liveRangeInfo),
          _depsInfo(depsInfo),
          _aliasInfo(aliasInfo),
          _scan(scan) {
}

void FeasibleMemoryScheduler::pushToStartTimeHeap(const HeapElement& elem) {
    _startTimeHeap.push_back(elem);
    std::push_heap(_startTimeHeap.begin(), _startTimeHeap.end(), MinHeapOrdering());
}

FeasibleMemoryScheduler::HeapElement FeasibleMemoryScheduler::popFromStartTimeHeap() {
    VPUX_THROW_UNLESS(!_startTimeHeap.empty(), "Tried to pop from empty _startTimeHeap");
    std::pop_heap(_startTimeHeap.begin(), _startTimeHeap.end(), MinHeapOrdering());
    HeapElement elem = _startTimeHeap.back();
    _startTimeHeap.pop_back();
    return elem;
}

void FeasibleMemoryScheduler::pushToCompletionTimeHeap(const HeapElement& elem) {
    _completionTimeHeap.push_back(elem);
    std::push_heap(_completionTimeHeap.begin(), _completionTimeHeap.end(), MinHeapOrdering());
}

FeasibleMemoryScheduler::HeapElement FeasibleMemoryScheduler::popFromCompletionTimeHeap() {
    VPUX_THROW_UNLESS(!_completionTimeHeap.empty(), "Tried to pop from empty _completionTimeHeap");
    std::pop_heap(_completionTimeHeap.begin(), _completionTimeHeap.end(), MinHeapOrdering());
    HeapElement elem = _completionTimeHeap.back();
    _completionTimeHeap.pop_back();
    return elem;
}

bool FeasibleMemoryScheduler::isDataOp(operationIdxType opIdx) {
    // Operations moving data to CMX are considered data ops. All others are
    // considered compute operations. This distinguishment is needed to balance
    // CMX memory space and not to fill CMX space with only data operations resulting
    // in not being able to fit the compute operation. Data operations will only be
    // scheduled when needed by the compute operation so that the CMX space can be
    // freed as soon as possible.
    auto op = _depsInfo.getExecuteOpAtIndex(opIdx);

    if (!op->hasAttr(IERT::IERTDialect::getExecutorAttrName())) {
        return false;
    }

    const auto executor = IERT::IERTDialect::getExecutor(op);
    if (executor.getLeafNameAttr() != VPU::ExecutorKindAttr::get(op->getContext(), VPU::ExecutorKind::DMA_NN)) {
        return false;
    }

    if (_outputOps.find(opIdx) != _outputOps.end()) {
        return false;
    }

    auto* bodyBlock = &op.body().front();
    if (op.getOperands().empty()) {
        for (auto& innerOp : bodyBlock->getOperations()) {
            for (const auto& operand : innerOp.getOperands()) {
                if (operand.getDefiningOp() == nullptr) {
                    // operation using function input
                    // input considered to be a compute operation
                    return false;
                }
            }
        }
    }

    for (auto& innerOp : bodyBlock->getOperations()) {
        IERT::CopyOp copyOp;
        // CopyOp can be placed directly in async exec op or wrapped with NCEClusterTiling
        if (auto nceClustOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(innerOp)) {
            copyOp = mlir::dyn_cast<IERT::CopyOp>(nceClustOp.getInnerTaskOp());
        } else {
            copyOp = mlir::dyn_cast<IERT::CopyOp>(innerOp);
        }

        if (copyOp) {
            // DMA from DDR to NN_CMX
            auto srcMemSpace = copyOp.input().getType().cast<vpux::NDTypeInterface>().getMemSpace();
            auto dstMemSpace = copyOp.output().getType().cast<vpux::NDTypeInterface>().getMemSpace();
            return (_memSpace == dstMemSpace && _memSpace != srcMemSpace);
        }
    }

    return false;
}

bool FeasibleMemoryScheduler::isNonComputeChainOp(operationIdxType opIdx) {
    // Currently only operations in the model which are not related to
    // processing network inputs are profiling related operations.
    auto op = _depsInfo.getExecuteOpAtIndex(opIdx);
    auto curTaskName = stringifyLocation(op->getLoc());
    if (curTaskName.find(PROFILING_CMX_2_DDR_OP_NAME) != std::string::npos) {
        return true;
    }

    return false;
}

bool FeasibleMemoryScheduler::isCopyOutOp(operationIdxType opIdx) {
    if (isDataOp(opIdx)) {
        return false;
    }

    auto op = _depsInfo.getExecuteOpAtIndex(opIdx);
    auto* bodyBlock = &op.body().front();
    for (auto& innerOp : bodyBlock->getOperations()) {
        IERT::CopyOp copyOp;
        // CopyOp can be placed directly in async exec op or wrapped with NCEClusterTiling
        if (auto nceClustOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(innerOp)) {
            copyOp = mlir::dyn_cast<IERT::CopyOp>(nceClustOp.getInnerTaskOp());
        } else {
            copyOp = mlir::dyn_cast<IERT::CopyOp>(innerOp);
        }

        if (copyOp) {
            auto dstMemSpace = copyOp.output().getType().cast<vpux::NDTypeInterface>().getMemSpace();
            return _memSpace != dstMemSpace;
        }
    }

    return false;
}

FeasibleMemoryScheduler::HeapElement const* FeasibleMemoryScheduler::topElementGen(ArrayRef<HeapElement> heap) const {
    return heap.empty() ? nullptr : &(heap.front());
}

SmallVector<FeasibleMemoryScheduler::HeapElement> FeasibleMemoryScheduler::popAllElementsAtThisTime(size_t time_step) {
    SmallVector<HeapElement> poppedOps;
    HeapElement const* topPtr = nullptr;
    while ((topPtr = topElementGen(_completionTimeHeap)) && topPtr->time_ == time_step) {
        poppedOps.push_back(popFromCompletionTimeHeap());
        std::push_heap(poppedOps.begin(), poppedOps.end(), MinHeapOrdering());
    }
    return poppedOps;
}

void FeasibleMemoryScheduler::unscheduleOp(const HeapElement& hElemet) {
    auto op = _depsInfo.getExecuteOpAtIndex(hElemet.op_);
    // free possible buffers, where this is the last user of the buffer
    const auto usedBufs = _liveRangeInfo.getUsedBuffers(op);
    for (auto buffer : usedBufs) {
        auto rootBuffers = _aliasInfo.getRoots(buffer);
        VPUX_THROW_UNLESS(rootBuffers.size() == 1, "Value '{0}' expected to have only one root. Got {1}", buffer,
                          rootBuffers.size());
        const auto rootBuffer = *rootBuffers.begin();
        if (_liveRangeInfo.eraseUser(rootBuffer, op) == 0) {
            _log.nest().trace("Mark buffer as dead, '{0}'", rootBuffer);
            _scan.handler().markAsDead(rootBuffer);
        }
    }
    _log.nest().trace("Free non alive buffers");
    _scan.freeNonAlive();

    // update consumers of op dependencies (consumed by this op)
    if (!hElemet.isImplicitWriteOp()) {
        for (auto dep : _depsInfo.getOpDeps(hElemet.op_)) {
            auto depOutput = _opOutputTable.find(dep);
            if (depOutput->second.active()) {
                depOutput->second.decrementConsumers();
            }
        }
        auto opOutput = _opOutputTable.find(hElemet.op_);
        if (opOutput->second.consumed()) {
            opOutput->second.changeStateToConsumed();
        }
    }
}

bool FeasibleMemoryScheduler::isComputeOpWithSomeActiveInputs(operationIdxType opIdx) {
    if (isDataOp(opIdx)) {
        return false;
    }

    // Get operation operands and find corresponding root buffers.
    // If any of such buffers is alive then input is active
    auto op = _depsInfo.getExecuteOpAtIndex(opIdx);
    for (const auto& operand : op.getOperands()) {
        if (!operand.getType().isa<mlir::async::ValueType>()) {
            continue;
        }
        auto rootBuffers = _aliasInfo.getRoots(operand);
        VPUX_THROW_UNLESS(rootBuffers.size() == 1, "Value '{0}' expected to have only one root. Got {1}", operand,
                          rootBuffers.size());
        const auto rootBuffer = *rootBuffers.begin();
        const auto type = rootBuffer.getType().cast<vpux::NDTypeInterface>();
        if (type.getMemSpace() != _memSpace) {
            continue;
        }
        if (_scan.handler().isAlive(rootBuffer)) {
            return true;
        }
    }

    return false;
}

vpux::AddressType FeasibleMemoryScheduler::calculateOpSize(operationIdxType opIdx) {
    // only use the output size
    vpux::AddressType opSize = 0;
    auto* bodyBlock = &_depsInfo.getExecuteOpAtIndex(opIdx).body().front();
    for (auto& op : bodyBlock->getOperations()) {
        if (mlir::isa<IERT::LayerOpInterface>(op)) {
            auto outputs = mlir::dyn_cast<IERT::LayerOpInterface>(op).getOutputs();
            for (const auto& output : outputs) {
                const auto type = output.getType().dyn_cast<vpux::NDTypeInterface>();
                if (type == nullptr || type.getMemSpace() != _memSpace) {
                    continue;
                }
                opSize += _scan.handler().getSize(output);
            }
            auto inputs = mlir::dyn_cast<IERT::LayerOpInterface>(op).getInputs();
            for (const auto& input : inputs) {
                const auto type = input.getType().dyn_cast<vpux::NDTypeInterface>();
                if (type == nullptr || type.getMemSpace() != _memSpace) {
                    continue;
                }
                opSize += _scan.handler().getSize(input);
            }
        }
    }
    return opSize;
}

void FeasibleMemoryScheduler::distributeReadyOps(llvm::ArrayRef<operationIdxType> readyOps) {
    // populate ready lists depending on op type/state
    _log.trace("Distribute new ready ops");
    _log = _log.nest();
    for (auto& readyOp : readyOps) {
        vpux::AddressType opSize = calculateOpSize(readyOp);
        auto pair = std::make_pair(readyOp, opSize);
        if (isDataOp(readyOp)) {
            VPUX_THROW_UNLESS(_readyDataOps.find(pair) == _readyDataOps.end(),
                              "Operation already in the ready data list '{0}'", readyOp);
            _readyDataOps.insert(std::make_pair(readyOp, opSize));
            _log.trace("Add to ready data ops '{0}'", readyOp);
            const auto newReadyOps = reduceInDegreeOfAdjacentOperations(readyOp);
            distributeReadyOps(newReadyOps);
        } else if (isNonComputeChainOp(readyOp)) {
            VPUX_THROW_UNLESS(_nonComputeChainOps.find(pair) == _nonComputeChainOps.end(),
                              "Operation already in non compute chain op list '{0}'", readyOp);
            _nonComputeChainOps.insert(std::make_pair(readyOp, opSize));
            _log.trace("Non compute chain op ready '{0}'", readyOp);
        } else if (isComputeOpWithSomeActiveInputs(readyOp)) {
            VPUX_THROW_UNLESS(_activeComputeOps.find(pair) == _activeComputeOps.end(),
                              "Operation already in active compute list '{0}'", readyOp);
            _activeComputeOps.insert(std::make_pair(readyOp, opSize));
            _log.trace("Add to active compute ops '{0}'", readyOp);
        } else {
            VPUX_THROW_UNLESS(_readyComputeOps.find(pair) == _readyComputeOps.end(),
                              "Operation already in ready compute list '{0}'", readyOp);
            _readyComputeOps.insert(std::make_pair(readyOp, opSize));
            _log.trace("Add to ready compute ops '{0}'", readyOp);
        }
    }
    _log = _log.unnest();
}

void FeasibleMemoryScheduler::unscheduleAllCompletingOpsAtNextEarliestTime() {
    // retrieve the latest time
    const HeapElement* completionTopPtr = topElementGen(_completionTimeHeap);
    VPUX_THROW_UNLESS(completionTopPtr != nullptr, "Got empty completionTopPtr");
    _currentTime = completionTopPtr->time_;
    _log.trace("Unscheduling ops at time: '{0}'", _currentTime);

    SmallVector<HeapElement> unscheduledOps = popAllElementsAtThisTime(_currentTime);
    SmallVector<operationIdxType> ready_ops = {};

    _log = _log.nest();
    for (auto& op : unscheduledOps) {
        auto opIdx = op.op_;
        _log.trace("Unscheduling '{0}'", opIdx);
        unscheduleOp(op);
        if (!isDataOp(opIdx) && op.isOriginalOp()) {
            // propagate through original compute ops, generate new ready ops
            auto newReadyOps = reduceInDegreeOfAdjacentOperations(opIdx);
            _log.nest().trace("Reduce consumer indegree");
            ready_ops.insert(ready_ops.end(), newReadyOps.begin(), newReadyOps.end());
        }
    }
    _log = _log.unnest();

    distributeReadyOps(ready_ops);
}

SmallVector<operationIdxType> FeasibleMemoryScheduler::reduceInDegreeOfAdjacentOperations(operationIdxType opIdx) {
    SmallVector<operationIdxType> zeroInDegreeOps;
    // reduce indegree (number of incoming edges) for consumers of ready data ops
    for (auto consumer : _depsInfo.getConsumerOps(opIdx)) {
        if (_inDegreeTable[consumer] < 2) {
            zeroInDegreeOps.push_back(consumer);
            _inDegreeTable.erase(consumer);
        } else {
            VPUX_THROW_UNLESS(_inDegreeTable[consumer] > 0, "Invalid indegree");
            _inDegreeTable[consumer]--;
        }
    }
    return zeroInDegreeOps;
}

void FeasibleMemoryScheduler::getReadyDataList() {
    _log.trace("Initial ready data list:");
    _log = _log.nest();
    // populate ready data ops
    for (auto& entry : _inDegreeTable) {
        if (entry.second == 0 && isDataOp(entry.first) && !isNonComputeChainOp(entry.first)) {
            vpux::AddressType opSize = calculateOpSize(entry.first);
            _readyDataOps.insert(std::make_pair(entry.first, opSize));
            _log.trace("Ready data op: '{0}'", entry.first);
            // reduce indegree of op consumers
            auto newOps = reduceInDegreeOfAdjacentOperations(entry.first);
            if (!newOps.empty()) {
                for (auto& op : newOps) {
                    VPUX_THROW_UNLESS(isNonComputeChainOp(op),
                                      "Not a non compute chain op unlocked - unsupported scenario");
                    opSize = calculateOpSize(op);
                    _nonComputeChainOps.insert(std::make_pair(op, opSize));
                    _log.trace("Ready non compute chain op: '{0}'", entry.first);
                }
            }
        }
    }
    _log = _log.unnest();
}

void FeasibleMemoryScheduler::getReadyComputeList() {
    _log.trace("Initial ready compute list:");
    _log = _log.nest();
    // populate ready compute ops
    for (auto& entry : _inDegreeTable) {
        if (entry.second == 0 && !isDataOp(entry.first) && !isNonComputeChainOp(entry.first)) {
            vpux::AddressType opSize = calculateOpSize(entry.first);
            _readyComputeOps.insert(std::make_pair(entry.first, opSize));
            _log.trace("Ready compute op: '{0}'", entry.first);
        }
    }
    _log = _log.unnest();
}

SmallVector<mlir::Value> FeasibleMemoryScheduler::sortUsedBuffers(mlir::DenseSet<mlir::Value>& operationBuffers) {
    SmallVector<std::pair<vpux::AddressType, mlir::Value>> buffersWithSize;
    for (auto& val : operationBuffers) {
        auto opSize = _scan.handler().getSize(val);
        for (auto user : val.getUsers()) {
            if (user->hasAttr("exceedingNNCMX")) {
                // allocate exceeding buffers first to not exceed NNCMX
                _log.trace("Re-ordering exceeding NNCMX buffer: '{0}'", val);
                opSize = std::numeric_limits<vpux::AddressType>::max();
            }
        }
        buffersWithSize.push_back(std::make_pair(opSize, val));
    }
    // sort based on size of buffer
    llvm::sort(buffersWithSize.begin(), buffersWithSize.end(),
               [](const std::pair<vpux::AddressType, mlir::Value>& val1,
                  const std::pair<vpux::AddressType, mlir::Value>& val2) {
                   // If size is the same use position in IR to have
                   // consistent order between executions
                   if (val1.first == val2.first) {
                       const auto parentOp = val1.second.getDefiningOp();
                       VPUX_THROW_UNLESS(parentOp != nullptr, "Block arguments are not supported");
                       return parentOp->isBeforeInBlock(val2.second.getDefiningOp());
                   }
                   return val1.first > val2.first;
               });
    // repopulate only with buffers
    SmallVector<mlir::Value> orderedBufs;
    for (auto& pair : buffersWithSize) {
        orderedBufs.push_back(pair.second);
    }
    return orderedBufs;
}

SmallVector<mlir::Value> FeasibleMemoryScheduler::getNonAliveBuffersUsedByOperation(operationIdxType opIdx) {
    // retrieve all buffers used by the op which are not alive
    auto op = _depsInfo.getExecuteOpAtIndex(opIdx);
    auto usedBuffs = _liveRangeInfo.getUsedBuffers(op);
    SmallVector<mlir::Value> operationBuffers;

    for (auto& buffer : usedBuffs) {
        auto rootBuffers = _aliasInfo.getRoots(buffer);
        VPUX_THROW_UNLESS(rootBuffers.size() == 1, "Value '{0}' expected to have only one root. Got {1}", buffer,
                          rootBuffers.size());
        const auto rootBuffer = *rootBuffers.begin();

        const auto type = rootBuffer.getType().cast<vpux::NDTypeInterface>();
        if (type.getMemSpace() != _memSpace || _scan.handler().isAlive(rootBuffer)) {
            continue;
        }
        operationBuffers.push_back(rootBuffer);
    }
    return operationBuffers;
}

mlir::DenseSet<operationIdxType> FeasibleMemoryScheduler::getNonEmptyOpDemandList(
        operationIdxType opIdx, llvm::ArrayRef<mlir::Value> neededBuffers) {
    // return all buffers of an op that require allocation
    mlir::DenseSet<operationIdxType> demandList;
    for (auto& dep : _depsInfo.getOpDeps(opIdx)) {
        if (_opOutputTable.find(dep) == _opOutputTable.end()) {
            demandList.insert(dep);
        } else if (_opOutputTable[dep].spilled()) {
            // in case of multpile output buffers, ensure the spilled buffer is required
            for (auto& buffer : neededBuffers) {
                if (_opWritingToBuffer.find(buffer) != _opWritingToBuffer.end()) {
                    auto writerOp = retrieveBufferWriter(buffer);
                    auto executeOpIdx = _depsInfo.getIndex(
                            writerOp->getBlock()->getParent()->getParentOfType<mlir::async::ExecuteOp>());
                    if (executeOpIdx == dep) {
                        demandList.insert(dep);
                    }
                }
            }
        }
    }
    return demandList;
}

bool FeasibleMemoryScheduler::isReadyComputeOperationSchedulable(operationIdxType opIdx) {
    // retrieve op demand list - input ops
    auto usedBuffers = getNonAliveBuffersUsedByOperation(opIdx);
    auto demandList = getNonEmptyOpDemandList(opIdx, usedBuffers);
    mlir::DenseSet<mlir::Value> buffersNeedingAllocation;

    // retrieve operation's buffers that need allocation
    for (auto val : getNonAliveBuffersUsedByOperation(opIdx)) {
        buffersNeedingAllocation.insert(val);
    }

    // retrieve operation input's buffers
    for (auto inputIdx : demandList) {
        for (auto val : getNonAliveBuffersUsedByOperation(inputIdx)) {
            buffersNeedingAllocation.insert(val);
        }
    }

    // sort to minimize fragmentation
    auto sortedBuffers = sortUsedBuffers(buffersNeedingAllocation);
    // are resources available and can be allocated
    auto canAlloc = _scan.canAlloc(sortedBuffers);

    if (!canAlloc) {
        // if failed possibly a case with exceeding NNCMX, need to re-order
        sortedBuffers = sortUsedBuffers(buffersNeedingAllocation);
        canAlloc = _scan.canAlloc(sortedBuffers);
    }

    return canAlloc;
}

void FeasibleMemoryScheduler::scheduleInputOpForComputeOp(operationIdxType inputIdx, size_t delay) {
    // schedule the dependency - Data op
    _log.nest().trace("Scheduling input for compute op:'{0}'", inputIdx);
    _opOutputTable.insert(std::make_pair(inputIdx, OpOutputInfo(EOpState::ACTIVE, _outDegreeTable[inputIdx])));
    pushToStartTimeHeap(HeapElement(inputIdx, _currentTime + delay, EOpType::ORIGINAL_OP));
}

void FeasibleMemoryScheduler::scheduleSpilledOpBuffer(operationIdxType inputIdx, mlir::Value* buffer) {
    // schedule the spilled dependency
    _log.nest().trace("Scheduling spilled op:'{0}'", inputIdx);
    auto _opOutput = _opOutputTable.find(inputIdx);
    if (!_opOutput->second.spillIdx_.empty()) {
        _opOutput->second.spillIdx_.erase(getOpBufferOutputIdx(inputIdx, *buffer));
    }

    (_opOutput->second).changeStateToActive();
    // also store the buffer spilled
    auto spilledReadBuffer = *buffer;
    pushToStartTimeHeap(HeapElement(inputIdx, _currentTime, EOpType::IMPLICIT_OP_READ, spilledReadBuffer));
}

size_t FeasibleMemoryScheduler::allocateBuffersAndInputOps(operationIdxType opIdx, Partitioner::Direction allocDir) {
    // retrieve op demand list - input ops
    auto usedBuffers = getNonAliveBuffersUsedByOperation(opIdx);
    auto demandList = getNonEmptyOpDemandList(opIdx, usedBuffers);
    size_t maxInputDelay = demandList.empty() ? 0 : 1;
    mlir::DenseSet<mlir::Value> buffersNeedingAllocation;

    // retrieve operation's buffers that need allocation
    for (auto& val : usedBuffers) {
        buffersNeedingAllocation.insert(val);
        _log.nest().trace("Mark buffer as alive, '{0}'", val);
        _scan.handler().markAsAlive(val);
        // special case for spilled reads
        if (_opWritingToBuffer.find(val) != _opWritingToBuffer.end()) {
            auto writerOp = retrieveBufferWriter(val);
            auto executeOpIdx =
                    _depsInfo.getIndex(writerOp->getBlock()->getParent()->getParentOfType<mlir::async::ExecuteOp>());
            demandList.erase(executeOpIdx);
            scheduleSpilledOpBuffer(executeOpIdx, &val);
            maxInputDelay = 1;
        }
    }

    // retrieve operation input's buffers
    for (auto inputIdx : demandList) {
        for (auto val : getNonAliveBuffersUsedByOperation(inputIdx)) {
            buffersNeedingAllocation.insert(val);
            _log.nest().trace("Mark input op buffer as alive, '{0}'", val);
            _scan.handler().markAsAlive(val);
            // check if non-alive input buffer was spilled
            if (_opWritingToBuffer.find(val) != _opWritingToBuffer.end()) {
                // if so schedule the required spill read for it
                auto writerOp = retrieveBufferWriter(val);
                auto executeOpIdx = _depsInfo.getIndex(
                        writerOp->getBlock()->getParent()->getParentOfType<mlir::async::ExecuteOp>());
                scheduleSpilledOpBuffer(executeOpIdx, &val);
                maxInputDelay = 2;
            }
        }
        scheduleInputOpForComputeOp(inputIdx, maxInputDelay - 1);
    }

    auto sortedBuffers = sortUsedBuffers(buffersNeedingAllocation);

    // allocate buffers using LinearScan
    _log.nest().trace("Allocate memory for the alive buffers");
    VPUX_THROW_UNLESS(_scan.alloc(sortedBuffers, false, allocDir), "Failed to statically allocate '{0}' memory",
                      _memSpace);

    // Check if any of operation input dependencies have been scheduled
    // in the same scheduler iteration. In such case delay might need to be adjusted
    // based on start time of its input dependencies
    size_t depOpsMaxTimeInStartHeap = 0;
    for (auto& dep : _depsInfo.getOpDeps(opIdx)) {
        auto depOpInStartHeap = std::find_if(_startTimeHeap.begin(), _startTimeHeap.end(), [&](HeapElement el) {
            return (dep == el.op_);
        });
        if (depOpInStartHeap != _startTimeHeap.end()) {
            if (depOpInStartHeap->time_ > depOpsMaxTimeInStartHeap) {
                depOpsMaxTimeInStartHeap = depOpInStartHeap->time_;
            }
        }
    }
    if (depOpsMaxTimeInStartHeap > 0) {
        if (_currentTime + maxInputDelay <= depOpsMaxTimeInStartHeap) {
            maxInputDelay = depOpsMaxTimeInStartHeap + 1 - _currentTime;
        }
    }

    return maxInputDelay;
}

size_t FeasibleMemoryScheduler::scheduleComputeOp(operationIdxType opIdx) {
    // Step 1: add to output result table
    _opOutputTable.insert(std::make_pair(opIdx, OpOutputInfo(EOpState::ACTIVE, _outDegreeTable[opIdx])));

    // Step 2: assign resources simultaneously
    auto maxInputDelay = allocateBuffersAndInputOps(opIdx);

    // TODO: case for inplace ops

    // check if any prefetched operations are unlocked by this op
    if (_prefetchEdges.find(opIdx) != _prefetchEdges.end()) {
        for (auto prefetchDataOp : _prefetchEdges[opIdx]) {
            _prefetchDataOps.insert(prefetchDataOp.first);
        }
    }

    // Step 3: schedule the compute op
    size_t opStartTime = _currentTime + maxInputDelay;
    pushToStartTimeHeap(HeapElement(opIdx, opStartTime, EOpType::ORIGINAL_OP));

    return opStartTime;
}

size_t FeasibleMemoryScheduler::scheduleAllPossibleReadyOpsAndUpdate() {
    SmallVector<std::pair<operationIdxType, vpux::AddressType>> scheduledActiveOps;
    SmallVector<std::pair<operationIdxType, vpux::AddressType>> scheduledReadyOps;
    SmallVector<std::pair<operationIdxType, vpux::AddressType>> scheduledNonComputeChainOps;
    auto computeOpStartTime = _currentTime;
    _log.trace("Scheduling all possible ready ops");
    _log = _log.nest();

    // TODO: heuristic for choosing next schedulable op
    // TODO: struct for active and ready ops with sort by priority

    // 1. schedule all possible copy out operations
    //      1-1. schedule active copy out ops
    for (auto& readyOp : _activeComputeOps) {
        if (isReadyComputeOperationSchedulable(readyOp.first) && isCopyOutOp(readyOp.first)) {
            _log.trace("Scheduling active copy-out op: '{0}'", readyOp.first);
            computeOpStartTime = std::max(computeOpStartTime, scheduleComputeOp(readyOp.first));
            scheduledActiveOps.push_back(readyOp);
        }
    }
    //      1-2. schedule ready copy out ops
    for (auto& readyOp : _readyComputeOps) {
        if (isReadyComputeOperationSchedulable(readyOp.first) && isCopyOutOp(readyOp.first)) {
            _log.trace("Scheduling ready copy-out op: '{0}'", readyOp.first);
            computeOpStartTime = std::max(computeOpStartTime, scheduleComputeOp(readyOp.first));
            scheduledReadyOps.push_back(readyOp);
        }
    }

    //      1-3. schedule operation not belonging to main network compute chain
    for (auto& readyOp : _nonComputeChainOps) {
        // Scheduling such operations can only happen once all input dependencies
        // (both data and compute ops) have already been executed. This is different
        // to standard compute op which as part of its scheduling can force scheduling
        // of needed data ops
        bool areDepsReady = true;
        for (auto& dep : _depsInfo.getOpDeps(readyOp.first)) {
            if (_opOutputTable.find(dep) == _opOutputTable.end()) {
                areDepsReady = false;
                break;
            }
        }
        if (areDepsReady && isReadyComputeOperationSchedulable(readyOp.first)) {
            _log.trace("Scheduling non compute chain op: '{0}'", readyOp.first);
            computeOpStartTime = std::max(computeOpStartTime, scheduleComputeOp(readyOp.first));
            scheduledNonComputeChainOps.push_back(readyOp);
        }
    }

    // 2. schedule a true compute operation
    bool trueComputeScheduled = false;
    //      2-1. schedule active op that fit in CMX
    for (auto& readyOp : _activeComputeOps) {
        if (isReadyComputeOperationSchedulable(readyOp.first) && !isCopyOutOp(readyOp.first) && !trueComputeScheduled) {
            _log.trace("Scheduling active compute op: '{0}'", readyOp.first);
            computeOpStartTime = std::max(computeOpStartTime, scheduleComputeOp(readyOp.first));
            scheduledActiveOps.push_back(readyOp);
            trueComputeScheduled = true;
        }
    }
    //      2-2. schedule ready compute op
    for (auto& readyOp : _readyComputeOps) {
        if (isReadyComputeOperationSchedulable(readyOp.first) && !isCopyOutOp(readyOp.first) && !trueComputeScheduled) {
            _log.trace("Scheduling ready compute op: '{0}'", readyOp.first);
            computeOpStartTime = std::max(computeOpStartTime, scheduleComputeOp(readyOp.first));
            scheduledReadyOps.push_back(readyOp);
            trueComputeScheduled = true;
        }
    }

    // 3. update ready lists by removing scheduled ops
    for (auto scheduledOp : scheduledActiveOps) {
        _activeComputeOps.erase(scheduledOp);
    }
    for (auto scheduledOp : scheduledReadyOps) {
        _readyComputeOps.erase(scheduledOp);
    }
    for (auto scheduledOp : scheduledNonComputeChainOps) {
        _nonComputeChainOps.erase(scheduledOp);
    }

    _log = _log.unnest();
    return computeOpStartTime;
}

void FeasibleMemoryScheduler::schedulePrefetchedDataOps(size_t computeOpStartTime) {
    std::set<operationIdxType> scheduledOps;
    for (auto prefetchDataOpIdx : _prefetchDataOps) {
        // if operation is ready
        auto inDegree = _inDegreeTable.find(prefetchDataOpIdx);
        if (inDegree == _inDegreeTable.end() || inDegree->second == 0) {
            // if op can fit in NNCMX
            if (isReadyComputeOperationSchedulable(prefetchDataOpIdx)) {
                scheduledOps.insert(prefetchDataOpIdx);

                // skip if operation already allocated
                if (getNonAliveBuffersUsedByOperation(prefetchDataOpIdx).empty()) {
                    continue;
                }

                // Step 1: add to output result table
                _opOutputTable.insert(std::make_pair(
                        prefetchDataOpIdx, OpOutputInfo(EOpState::ACTIVE, _outDegreeTable[prefetchDataOpIdx])));

                // Step 2: assign resources simultaneously
                allocateBuffersAndInputOps(prefetchDataOpIdx, Partitioner::Direction::Down);

                // Step 3: schedule the prefetched data op
                _log.nest().trace("Scheduling prefetched data op:'{0}'", prefetchDataOpIdx);
                pushToStartTimeHeap(
                        HeapElement(prefetchDataOpIdx, computeOpStartTime, EOpType::ORIGINAL_PREFETCHED_OP));
            }
        }
    }
    for (auto scheduledOpIdx : scheduledOps) {
        _prefetchDataOps.erase(scheduledOpIdx);
    }

    // remove the not scheduled prefetch operations from the ready list
    _prefetchDataOps.clear();
}

void FeasibleMemoryScheduler::evictActiveOp(EvictionCandidate evictionCandidate) {
    auto opOutput = _opOutputTable.find(evictionCandidate.bufferWriterIdx_);
    VPUX_THROW_UNLESS(opOutput != _opOutputTable.end(), "Attempt to evict a non-scheduled operation");

    if (evictionCandidate.outputIdx_ != 0 || !opOutput->second.spillIdx_.empty()) {
        // MultiViewOp case for spilling with multiple output buffers
        VPUX_THROW_UNLESS(
                opOutput->second.spillIdx_.find(evictionCandidate.outputIdx_) == opOutput->second.spillIdx_.end(),
                "Attempt to evict the same buffer twice");
        opOutput->second.spillIdx_.insert(evictionCandidate.outputIdx_);
    } else {
        VPUX_THROW_UNLESS(opOutput->second.active(), "Attempt to evict a non active operation");
    }

    // update _opOutputTable, as consumers increse
    opOutput->second.changeStateToSpilled();
    opOutput->second.outstandingConsumers_++;

    // increment consumers of dependencies due to spilled op
    for (auto dep : _depsInfo.getOpDeps(evictionCandidate.bufferWriterIdx_)) {
        auto depOutput = _opOutputTable.find(dep);
        depOutput->second.incrementConsumers();
    }

    _log.nest().trace("Mark buffer as dead, '{0}'", evictionCandidate.buffer_);
    _scan.handler().markAsDead(evictionCandidate.buffer_);

    _log.nest().trace("Free non alive buffers");
    _scan.freeNonAlive();
}

IERT::LayerOpInterface FeasibleMemoryScheduler::retrieveBufferWriter(mlir::Value buffer) {
    const auto valIt = _opWritingToBuffer.find(buffer);
    VPUX_THROW_UNLESS(valIt != _opWritingToBuffer.end(), "Buffer not scheduled yet, invalid spill candidate");
    return valIt->second;
}

size_t FeasibleMemoryScheduler::evictionPriority(mlir::Value buffer) {
    // TODO: EISW-21936 add other conditions such as:
    // pipelined, multiple outdegree (prefetch)

    // Eviction priority (highest evicted first):
    // (0) - buffers which are CMX Concatable
    // (1) - timestamp op buffers
    // (2) - buffers which are result of computeOp
    // (3) - buffers which are result of dataOp
    // (4) - buffers which are not going to be used by any active/ready compute ops

    auto writerOp = retrieveBufferWriter(buffer);
    auto executeOp = writerOp->getBlock()->getParent()->getParentOfType<mlir::async::ExecuteOp>();
    bool isBufferUsedByReadyOp = false;
    bool cmxConcatable = false;

    for (auto& active : _activeComputeOps) {
        auto op = _depsInfo.getExecuteOpAtIndex(active.first).getOperation();

        if (_liveRangeInfo.isBufferUsedByOp(buffer, op)) {
            isBufferUsedByReadyOp = true;
            break;
        }
    }

    if (!isBufferUsedByReadyOp) {
        for (auto& active : _readyComputeOps) {
            auto op = _depsInfo.getExecuteOpAtIndex(active.first).getOperation();

            if (_liveRangeInfo.isBufferUsedByOp(buffer, op)) {
                isBufferUsedByReadyOp = true;
                break;
            }
        }
    }

    for (auto bufferUser : buffer.getUsers()) {
        if (mlir::isa<IERT::ConcatViewOp>(bufferUser)) {
            cmxConcatable = true;
        }
    }

    if (cmxConcatable) {
        return 0;
    } else if (mlir::isa<IERT::TimestampOp>(writerOp)) {
        return 1;
    } else if (!isBufferUsedByReadyOp) {
        return 4;
    } else if (isDataOp(_depsInfo.getIndex(executeOp))) {
        return 3;
    } else {
        return 2;
    }
}

size_t FeasibleMemoryScheduler::getOpBufferOutputIdx(operationIdxType opIdx, mlir::Value buffer) {
    size_t outputIdx = 0;

    // Get asyncExecOp result corresponding to given buffer
    auto asyncExecOp = _depsInfo.getExecuteOpAtIndex(opIdx);
    mlir::Value asyncExecOpResult;
    for (auto res : asyncExecOp.results()) {
        const auto rootBuffers = _aliasInfo.getRoots(res);
        VPUX_THROW_UNLESS(rootBuffers.size() == 1, "Value '{0}' expected to have only one root. Got {1}", res,
                          rootBuffers.size());
        const auto rootBuffer = *rootBuffers.begin();
        if (rootBuffer == buffer) {
            asyncExecOpResult = res;
        }
    }

    VPUX_THROW_UNLESS(asyncExecOpResult != nullptr,
                      "Unable to find async.execOp (opIdx - '{0}') result corresponding to buffer '{1}'", opIdx,
                      buffer);

    // Get asyncExecOp result index
    for (size_t idx = 0; idx < asyncExecOp->getNumResults(); idx++) {
        auto resultAtIdx = asyncExecOp->getResult(static_cast<unsigned int>(idx));
        if (resultAtIdx.getType().isa<mlir::async::ValueType>()) {
            if (resultAtIdx == asyncExecOpResult) {
                break;
            }
            outputIdx++;
        }
    }

    return outputIdx;
}

FeasibleMemoryScheduler::EvictionCandidate FeasibleMemoryScheduler::chooseCandidateForEviction(
        mlir::DenseSet<mlir::Value> aliveBuffers) {
    // sort buffers using eviction priority
    std::set<EvictionCandidate, EvictionPriority> evictionCandidates;
    for (const auto& buffer : aliveBuffers) {
        auto rootBuffers = _aliasInfo.getRoots(buffer);
        VPUX_THROW_UNLESS(rootBuffers.size() == 1, "Value '{0}' expected to have only one root. Got {1}", buffer,
                          rootBuffers.size());
        const auto rootBuffer = *rootBuffers.begin();
        auto writerOp = retrieveBufferWriter(rootBuffer);
        auto executeOpIdx =
                _depsInfo.getIndex(writerOp->getBlock()->getParent()->getParentOfType<mlir::async::ExecuteOp>());
        size_t priority = evictionPriority(rootBuffer);
        size_t size = _scan.handler().getSize(rootBuffer);
        // in special case of multiple output buffers store output idx
        auto outputIdx = getOpBufferOutputIdx(executeOpIdx, rootBuffer);
        evictionCandidates.insert(EvictionCandidate(priority, size, executeOpIdx, outputIdx, rootBuffer));
    }

    // first element has the smallest priority
    return *evictionCandidates.begin();
}

void FeasibleMemoryScheduler::forceScheduleActiveOpEviction() {
    // retrieve the alive buffers
    auto aliveBuffers = _scan.handler().getAliveValues();
    VPUX_THROW_UNLESS(!aliveBuffers.empty(), "Failed, nothing to spill");

    // select a candidate op to be spilled
    auto evictionCandidate = chooseCandidateForEviction(aliveBuffers);
    _log.nest().trace("Candidate selected for eviction {0}", evictionCandidate.bufferWriterIdx_);

    // free the memory space by freeing the op output buffer
    evictActiveOp(evictionCandidate);
    _log.nest().trace("Candidate evicted and spilled");

    // update _activeComputeOps some may no longer be active
    auto consumers = _depsInfo.getConsumerOps(evictionCandidate.bufferWriterIdx_);
    std::set<std::pair<operationIdxType, vpux::AddressType>> noLongerActive = {};
    for (auto& active : _activeComputeOps) {
        for (auto consumer : consumers) {
            if (active.first == consumer) {
                noLongerActive.insert(active);
            }
        }
    }
    for (auto notActive : noLongerActive) {
        _activeComputeOps.erase(notActive);
        _readyComputeOps.insert(notActive);
    }

    // add with a spilled write state
    pushToStartTimeHeap(HeapElement(evictionCandidate.bufferWriterIdx_, _currentTime, EOpType::IMPLICIT_OP_WRITE,
                                    evictionCandidate.buffer_));
}

void FeasibleMemoryScheduler::populateScheduledOps(HeapElement& scheduledOp) {
    SmallVector<IntervalInfo> outputIntervals;
    SmallVector<IntervalInfo> inputIntervals;
    // store scheduled information
    if (scheduledOp.isImplicitWriteOp()) {
        // special case for a spill write with deallocation
        IntervalInfo interval;
        auto output = scheduledOp.spillBuffer_;
        // retrieve and store operation addresses
        interval.begin_ = checked_cast<size_t>(_scan.handler().getAddress(output));
        interval.end_ = interval.begin_ + checked_cast<size_t>(_scan.handler().getSize(output));
        interval.buffer_ = output;
        // SPILL WRITE has only input resource
        inputIntervals.push_back(interval);
        // deallocate only after addresses stored
        _scan.handler().deallocate(output);
    } else {
        // retrieve interval information, operation can have multiple output buffers
        auto* bodyBlock = &_depsInfo.getExecuteOpAtIndex(scheduledOp.op_).body().front();
        for (auto& op : bodyBlock->getOperations()) {
            if (mlir::isa<IERT::LayerOpInterface>(op)) {
                auto layerOp = mlir::dyn_cast<IERT::LayerOpInterface>(op);

                if (scheduledOp.isOriginalOp()) {
                    // Track input resources. SPILL READ has only output resource
                    mlir::DenseSet<mlir::Value> inputBuffers;
                    for (auto input : layerOp.getInputs()) {
                        const auto type = input.getType().dyn_cast<mlir::MemRefType>();
                        if (type == nullptr || type.getMemorySpace() != _memSpace) {
                            continue;
                        }

                        // Find the root buffer for a given output as output of an operation
                        // doesn't have to point directly to result of memref.alloc (e.g. might
                        // be a result of SubView).
                        const auto rootBuffers = _aliasInfo.getRoots(input);
                        VPUX_THROW_UNLESS(rootBuffers.size() == 1,
                                          "Value '{0}' expected to have only one root. Got {1}", input,
                                          rootBuffers.size());
                        const auto rootBuffer = *rootBuffers.begin();

                        if (_opWritingToBuffer.find(rootBuffer) == _opWritingToBuffer.end()) {
                            continue;
                        }

                        if (inputBuffers.find(rootBuffer) != inputBuffers.end()) {
                            continue;
                        }
                        inputBuffers.insert(rootBuffer);

                        IntervalInfo interval;
                        // retrieve and store operation addresses
                        interval.begin_ = checked_cast<size_t>(_scan.handler().getAddress(rootBuffer));
                        interval.end_ = interval.begin_ + checked_cast<size_t>(_scan.handler().getSize(rootBuffer));
                        interval.buffer_ = rootBuffer;
                        inputIntervals.push_back(interval);
                    }
                }

                // Track output resources
                for (auto output : layerOp.getOutputs()) {
                    const auto type = output.getType().dyn_cast<vpux::NDTypeInterface>();
                    if (type == nullptr || type.getMemSpace() != _memSpace) {
                        continue;
                    }

                    // Find the root buffer for a given output as output of an operation
                    // doesn't have to point directly to result of memref.alloc (e.g. might
                    // be a result of SubView).
                    const auto rootBuffers = _aliasInfo.getRoots(output);
                    VPUX_THROW_UNLESS(rootBuffers.size() == 1, "Value '{0}' expected to have only one root. Got {1}",
                                      output, rootBuffers.size());
                    const auto rootBuffer = *rootBuffers.begin();

                    // in case of spill only allocate the spill buffer
                    if (!scheduledOp.isOriginalOp() && rootBuffer != scheduledOp.spillBuffer_) {
                        continue;
                    }
                    IntervalInfo interval;
                    // store operation writing to the buffer
                    _opWritingToBuffer[rootBuffer] = layerOp;
                    // retrieve and store operation addresses
                    interval.begin_ = checked_cast<size_t>(_scan.handler().getAddress(rootBuffer));
                    interval.end_ = interval.begin_ + checked_cast<size_t>(_scan.handler().getSize(rootBuffer));
                    interval.buffer_ = rootBuffer;
                    outputIntervals.push_back(interval);
                }
            }
        }
    }
    // populate the struct fields
    ScheduledOpInfo scheduled;
    scheduled.op_ = scheduledOp.op_;
    scheduled.opType_ = scheduledOp.opType_;
    scheduled.time_ = scheduledOp.time_;
    scheduled.outputResourceInfo_ = outputIntervals;
    scheduled.inputResourceInfo_ = inputIntervals;
    scheduled.isDataOp_ = isDataOp(scheduledOp.op_);
    scheduled.freeCmx_ = _scan.totalFreeSize();
    _scheduledOps.push_back(scheduled);
}

void FeasibleMemoryScheduler::clearLists() {
    _readyComputeOps.clear();   // ready operations with no active input
    _readyDataOps.clear();      // ready data inputs (->CMX)
    _activeComputeOps.clear();  // compute operations with at least one active input
}

bool FeasibleMemoryScheduler::init() {
    _log.trace("Feasible Memory Scheduler init()");
    _currentTime = 1;
    _depsInfo.buildConsMap();

    // compute op in/out degree
    _inDegreeTable = _depsInfo.calculateOpInDegreeTable();
    _outDegreeTable = _depsInfo.calculateOpOutDegreeTable();

    // retrieve output ops (ops with no out-degree)
    for (auto& entry : _outDegreeTable) {
        if (entry.second == 0) {
            _outputOps.insert(entry.first);
        }
    }

    clearLists();
    // TODO: check if input is dag
    getReadyDataList();
    getReadyComputeList();
    scheduleAllPossibleReadyOpsAndUpdate();
    nextSchedulableOp();

    return true;
}

void FeasibleMemoryScheduler::nextSchedulableOp() {
    // scheduling loop, loop until all output ops are scheduled
    while (!_outputOps.empty()) {
        // choose the minimum time from start time and completion time heaps
        // to schedule the earliest possible operation
        const HeapElement* start_top_ptr = topElementGen(_startTimeHeap);
        const HeapElement* completionTopPtr = topElementGen(_completionTimeHeap);

        _log.trace("Choose the min from start time and completion time heaps");
        bool pop_from_start_heap = (start_top_ptr != nullptr) &&
                                   (completionTopPtr == nullptr || (start_top_ptr->time_ < completionTopPtr->time_));

        if (pop_from_start_heap) {
            _log.trace("Popping from start time heap");
            // schedule first op in heap
            HeapElement firstOp = popFromStartTimeHeap();
            _currentTime = firstOp.time_;
            // add to output table
            populateScheduledOps(firstOp);
            // move to completion time heap
            pushToCompletionTimeHeap(HeapElement(firstOp.op_, _currentTime + 1, firstOp.opType_));
            _log.trace("Scheduled op: '{0}'", firstOp.op_);
            // decrease outputs ops if output op scheduled
            if (_outputOps.find(firstOp.op_) != _outputOps.end()) {
                _outputOps.erase(firstOp.op_);
            }
        } else {
            do {
                _log.trace("Popping from completion time heap");
                // unschedule operations, and propagate through the graph by creating new ready ops
                unscheduleAllCompletingOpsAtNextEarliestTime();
                // with new ready ops created try to schedule new ops
                auto computeOpStartTime = scheduleAllPossibleReadyOpsAndUpdate();
                schedulePrefetchedDataOps(computeOpStartTime);
            } while (!_completionTimeHeap.empty() && _startTimeHeap.empty());

            if (_startTimeHeap.empty()) {
                // unable to schedule an operation, perform spill
                _log.trace("Unable to schedule an operation, forcing dynamic spill");
                forceScheduleActiveOpEviction();
            }
        }
    }
}

SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo> FeasibleMemoryScheduler::generateSchedule(
        prefetchMap prefetchEdges) {
    // iteration with prefetching edges
    if (!prefetchEdges.empty()) {
        _scan.handler().markAllBuffersAsDead();
        _prefetchEdges = prefetchEdges;
    }

    // start the memory scheduler
    init();

    // TODO: save schedule from _scheduledOps to file

    _log.trace("Generated Schedule");
    _log = _log.nest();
    for (const auto& op : _scheduledOps) {
        std::string inputResourceInfo = "<none>";
        std::string outputResourceInfo = "<none>";
        if (op.hasActiveInputResource()) {
            inputResourceInfo = "";
            for (size_t resourceIdx = 0; resourceIdx < op.numOfInputResources(); resourceIdx++) {
                if (op.isActiveInputResource(resourceIdx)) {
                    inputResourceInfo +=
                            "[" + std::to_string(op.beginInputResource(resourceIdx)) + " " +
                            std::to_string(op.endInputResource(resourceIdx)) + "] size = " +
                            std::to_string((op.endInputResource(resourceIdx) - op.beginInputResource(resourceIdx))) +
                            ", ";
                }
            }
        }
        if (op.hasActiveOutputResource()) {
            outputResourceInfo = "";
            for (size_t resourceIdx = 0; resourceIdx < op.numOfOutputResources(); resourceIdx++) {
                if (op.isActiveOutputResource(resourceIdx)) {
                    outputResourceInfo +=
                            "[" + std::to_string(op.beginOutputResource(resourceIdx)) + " " +
                            std::to_string(op.endOutputResource(resourceIdx)) + "] size = " +
                            std::to_string((op.endOutputResource(resourceIdx) - op.beginOutputResource(resourceIdx))) +
                            ", ";
                }
            }
        }

        _log.trace("op = '{0}'\t type = '{1}'\t time = '{2}'\t inputs = '{3}' outputs = '{4}'", op.op_, op.opTypeName(),
                   op.time_, inputResourceInfo, outputResourceInfo);
    }
    _log = _log.unnest();

    return _scheduledOps;
}
