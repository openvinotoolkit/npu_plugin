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

#include "vpux/compiler/utils/attributes.hpp"

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

    uint32_t numUnits;
    const auto executor = IERT::IERTDialect::getExecutor(op, numUnits);
    if (executor != VPUIP::DMAEngineAttr::get(op->getContext(), VPUIP::DMAEngine::DMA_NN)) {
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
        if (mlir::isa<IERT::CopyOp>(innerOp)) {
            if (auto copyOp = mlir::dyn_cast<IERT::CopyOp>(innerOp)) {
                // DMA from DDR to NN_CMX
                auto srcMemSpace = copyOp.input().getType().dyn_cast<mlir::MemRefType>().getMemorySpace();
                auto dstMemSpace = copyOp.output().getType().dyn_cast<mlir::MemRefType>().getMemorySpace();
                return (_memSpace == dstMemSpace && _memSpace != srcMemSpace);
            }
        }
    }

    return false;
}

FeasibleMemoryScheduler::HeapElement const* FeasibleMemoryScheduler::topElementGen(ArrayRef<HeapElement> heap) const {
    return heap.empty() ? nullptr : &(heap.front());
}

llvm::SmallVector<FeasibleMemoryScheduler::HeapElement> FeasibleMemoryScheduler::popAllElementsAtThisTime(
        size_t time_step) {
    llvm::SmallVector<HeapElement> poppedOps;
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
    for (auto val : usedBufs) {
        if (_liveRangeInfo.eraseUser(val, op) == 0) {
            _log.nest().trace("Mark buffer as dead, '{0}'", val);
            _scan.handler().markAsDead(val);
        }
    }
    _log.nest().trace("Free non alive buffers");
    _scan.freeNonAlive();

    // update consumers of op dependencies (consumed by this op)
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

bool FeasibleMemoryScheduler::isComputeOpWithSomeActiveInputs(operationIdxType opIdx) {
    if (isDataOp(opIdx)) {
        return false;
    }
    auto opAllDeps = _depsInfo.getOpDeps(opIdx);
    auto opDeps = getSortedBuffers(opIdx);
    // number of buffers needing allocation smaller than number of buffers used by the op
    return opAllDeps.size() > opDeps.size();
}

vpux::AddressType FeasibleMemoryScheduler::calculateOpSize(operationIdxType opIdx) {
    // only use the output size
    vpux::AddressType opSize = 0;
    auto* bodyBlock = &_depsInfo.getExecuteOpAtIndex(opIdx).body().front();
    for (auto& op : bodyBlock->getOperations()) {
        if (mlir::isa<IERT::LayerOpInterface>(op)) {
            auto outputs = mlir::dyn_cast<IERT::LayerOpInterface>(op).getOutputs();
            for (const auto& output : outputs) {
                const auto type = output.getType().dyn_cast<mlir::MemRefType>();
                if (type == nullptr || type.getMemorySpace() != _memSpace) {
                    continue;
                }
                opSize += _scan.handler().getSize(output);
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
                              "Operation already in the ready data list");
            _readyDataOps.insert(std::make_pair(readyOp, opSize));
            _log.trace("Add to ready data ops '{0}'", readyOp);
            const auto newReadyOps = reduceInDegreeOfAdjacentOperations(readyOp);
            distributeReadyOps(newReadyOps);
        } else if (isComputeOpWithSomeActiveInputs(readyOp)) {
            VPUX_THROW_UNLESS(_activeComputeOps.find(pair) == _activeComputeOps.end(),
                              "Operation already in active compute list");
            _activeComputeOps.insert(std::make_pair(readyOp, opSize));
            _log.trace("Add to active compute ops '{0}'", readyOp);
        } else {
            VPUX_THROW_UNLESS(_readyComputeOps.find(pair) == _readyComputeOps.end(),
                              "Operation already in ready compute list");
            _readyComputeOps.insert(std::make_pair(readyOp, opSize));
            _log.trace("Add to ready compute ops '{0}'", readyOp);
        }
    }
    _log = _log.unnest();
}

void FeasibleMemoryScheduler::unscheduleAllCompletingOpsAtNextEarliestTime() {
    // retrieve the latest time
    const HeapElement* completionTopPtr = topElementGen(_completionTimeHeap);
    _currentTime = completionTopPtr->time_;
    _log.trace("Unscheduling ops at time: '{0}'", _currentTime);

    llvm::SmallVector<HeapElement> unscheduledOps = popAllElementsAtThisTime(_currentTime);
    llvm::SmallVector<operationIdxType> ready_ops = {};

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

llvm::SmallVector<operationIdxType> FeasibleMemoryScheduler::reduceInDegreeOfAdjacentOperations(
        operationIdxType opIdx) {
    llvm::SmallVector<operationIdxType> zeroInDegreeOps;
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
        if (entry.second == 0 && isDataOp(entry.first)) {
            vpux::AddressType opSize = calculateOpSize(entry.first);
            _readyDataOps.insert(std::make_pair(entry.first, opSize));
            _log.trace("Ready data op: '{0}'", entry.first);
            // reduce indegree of op consumers
            reduceInDegreeOfAdjacentOperations(entry.first);
        }
    }
    _log = _log.unnest();
}

void FeasibleMemoryScheduler::getReadyComputeList() {
    _log.trace("Initial ready compute list:");
    _log = _log.nest();
    // populate ready compute ops
    for (auto& entry : _inDegreeTable) {
        if (entry.second == 0 && !isDataOp(entry.first)) {
            vpux::AddressType opSize = calculateOpSize(entry.first);
            _readyComputeOps.insert(std::make_pair(entry.first, opSize));
            _log.trace("Ready compute op: '{0}'", entry.first);
        }
    }
    _log = _log.unnest();
}

SmallVector<mlir::Value> FeasibleMemoryScheduler::getSortedBuffers(operationIdxType opIdx) {
    // retrieve all buffers used bu the op
    auto op = _depsInfo.getExecuteOpAtIndex(opIdx);
    auto usedBuffs = _liveRangeInfo.getUsedBuffers(op);
    SmallVector<std::pair<vpux::AddressType, mlir::Value>> newBufs;
    for (auto& val : usedBuffs) {
        const auto type = val.getType().cast<mlir::MemRefType>();
        if (type.getMemorySpace() != _memSpace || _scan.handler().isAlive(val)) {
            continue;
        }
        auto size = _scan.handler().getSize(val);
        newBufs.push_back(std::make_pair(size, val));
    }
    // sort based on size of buffer
    llvm::sort(newBufs.begin(), newBufs.end(),
               [](const std::pair<vpux::AddressType, mlir::Value>& val1,
                  const std::pair<vpux::AddressType, mlir::Value>& val2) {
                   return val1.first > val2.first;
               });
    SmallVector<mlir::Value> orderedBufs;
    for (auto& pair : newBufs) {
        orderedBufs.push_back(pair.second);
    }
    return orderedBufs;
}

mlir::DenseSet<operationIdxType> FeasibleMemoryScheduler::getNonEmptyOpDemandList(operationIdxType opIdx) {
    // return all buffers of an op that require allocation
    mlir::DenseSet<operationIdxType> demandList;
    for (auto& dep : _depsInfo.getOpDeps(opIdx)) {
        if (_opOutputTable.find(dep) == _opOutputTable.end() || _opOutputTable[dep].spilled()) {
            demandList.insert(dep);
        }
    }
    return demandList;
}

bool FeasibleMemoryScheduler::isReadyComputeOperationSchedulable(operationIdxType opIdx) {
    // are resources available and can be allocated
    auto sortedBuffers = getSortedBuffers(opIdx);
    return _scan.canAlloc(sortedBuffers);
}

void FeasibleMemoryScheduler::scheduleInputOpForComputeOp(operationIdxType inputIdx) {
    // schedule the dependency - Data op
    _log.nest().trace("Scheduling input for compute op:'{0}'", inputIdx);
    EOpType opType = EOpType::ORIGINAL_OP;
    _opOutputTable.insert(std::make_pair(inputIdx, OpOutputInfo(EOpState::ACTIVE, _outDegreeTable[inputIdx])));
    pushToStartTimeHeap(HeapElement(inputIdx, _currentTime, opType));
}

void FeasibleMemoryScheduler::scheduleSpilledInputOpForComputeOp(operationIdxType inputIdx, mlir::Value* buffer) {
    // schedule the spilled dependency
    _log.nest().trace("Scheduling spilled input for compute op:'{0}'", inputIdx);
    auto _opOutput = _opOutputTable.find(inputIdx);
    if (!_opOutput->second.spillIdx_.empty()) {
        _opOutput->second.spillIdx_.erase(getOpBufferOutputIdx(inputIdx, *buffer));
    }

    (_opOutput->second).changeStateToActive();
    EOpType opType = EOpType::IMPLICIT_OP_READ;
    // also store the buffer spilled
    auto spilledReadBuffer = *buffer;
    pushToStartTimeHeap(HeapElement(inputIdx, _currentTime, opType, spilledReadBuffer));
}

size_t FeasibleMemoryScheduler::allocateBuffersAndInputOps(operationIdxType opIdx,
                                                           SmallVector<mlir::Value>& sortedBuffers) {
    // retrieve op demand list - input ops
    auto demandList = getNonEmptyOpDemandList(opIdx);
    size_t maxInputDelay = demandList.empty() ? 0 : 1;

    // retrieve buffers that need allocation
    SmallVector<mlir::Value> buffersNeedingAllocation;
    for (auto& val : sortedBuffers) {
        const auto type = val.getType().cast<mlir::MemRefType>();
        if (type.getMemorySpace() != _memSpace || _scan.handler().isAlive(val)) {
            continue;
        }
        _log.nest().trace("Mark buffer as alive, '{0}'", val);
        _scan.handler().markAsAlive(val);
        buffersNeedingAllocation.push_back(val);
        // special case for spilled reads
        if (_opWritingToBuffer.find(val) != _opWritingToBuffer.end()) {
            auto bufferWriter = retrieveBufferWriter(val);
            demandList.erase(bufferWriter);
            scheduleSpilledInputOpForComputeOp(bufferWriter, &val);
        }
    }

    // schedule all not spilled inputs
    for (auto inputIdx : demandList) {
        scheduleInputOpForComputeOp(inputIdx);
    }

    // allocate buffers using LinearScan
    _log.nest().trace("Allocate memory for the alive buffers");
    VPUX_THROW_UNLESS(_scan.alloc(buffersNeedingAllocation, /*allowSpills*/ false),
                      "Failed to statically allocate '{0}' memory", _memSpace);

    return maxInputDelay;
}

void FeasibleMemoryScheduler::scheduleComputeOp(operationIdxType opIdx) {
    // Step 1: add to output result table
    _opOutputTable.insert(std::make_pair(opIdx, OpOutputInfo(EOpState::ACTIVE, _outDegreeTable[opIdx])));

    // Step 2: assign resources simultaneously
    auto sortedBuffers = getSortedBuffers(opIdx);
    auto maxInputDelay = allocateBuffersAndInputOps(opIdx, sortedBuffers);

    // TODO: case for inplace ops

    // Step 3: schedule the compute op
    size_t opStartTime = _currentTime + maxInputDelay;
    pushToStartTimeHeap(HeapElement(opIdx, opStartTime, EOpType::ORIGINAL_OP));
}

void FeasibleMemoryScheduler::scheduleAllPossibleReadyOpsAndUpdate(
        std::set<std::pair<operationIdxType, vpux::AddressType>, SizeSort>& readyList) {
    SmallVector<std::pair<operationIdxType, vpux::AddressType>> scheduledOps;
    _log.trace("Scheduling all possible ready ops");
    _log = _log.nest();
    // schedule ops that fit in CMX
    for (auto& readyOp : readyList) {
        if (isReadyComputeOperationSchedulable(readyOp.first)) {
            _log.trace("Scheduling ready op: '{0}'", readyOp.first);
            scheduleComputeOp(readyOp.first);
            scheduledOps.push_back(readyOp);
        }
    }
    _log = _log.unnest();
    // update ready lists by removing scheduled ops
    for (auto scheduledOp : scheduledOps) {
        readyList.erase(scheduledOp);
    }
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

operationIdxType FeasibleMemoryScheduler::retrieveBufferWriter(mlir::Value buffer) {
    const auto valIt = _opWritingToBuffer.find(buffer);
    VPUX_THROW_UNLESS(valIt != _opWritingToBuffer.end(), "Buffer not scheduled yet, invalid spill candidate");
    return valIt->second;
}

size_t FeasibleMemoryScheduler::evictionPriority(mlir::Value buffer) {
    // TODO: EISW-21937 add other conditions such as:
    // cmx concatable, pipelined, multiple outdegree (prefetch)

    // Eviction priority:
    // HIGH (0)   - buffers which are not going to be used by any active/ready compute ops
    // MEDIUM (1) - buffers which are result of dataOp
    // LOW (2)    - all others
    bool isBufferUsedByReadyOp = false;
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

    if (!isBufferUsedByReadyOp) {
        return 0;
    } else if (isDataOp(retrieveBufferWriter(buffer))) {
        return 1;
    } else {
        return 2;
    }
}

size_t FeasibleMemoryScheduler::getOpBufferOutputIdx(operationIdxType opIdx, mlir::Value buffer) {
    size_t outputIdx = 0;

    // Get AsyncExecOp result corresponding to given buffer
    auto asyncExecOp = _depsInfo.getExecuteOpAtIndex(opIdx);
    mlir::Value asyncExecOpResult;
    for (auto bufferAlias : _aliasInfo.getAllAliases(buffer)) {
        if (bufferAlias.getType().isa<mlir::async::ValueType>() && bufferAlias.getDefiningOp() == asyncExecOp) {
            asyncExecOpResult = bufferAlias;
            break;
        }
    }

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
        operationIdxType bufferWriterIdx = retrieveBufferWriter(buffer);
        size_t priority = evictionPriority(buffer);
        size_t size = _scan.handler().getSize(buffer);
        // in special case of multiple output buffers store output idx
        auto outputIdx = getOpBufferOutputIdx(bufferWriterIdx, buffer);
        evictionCandidates.insert(EvictionCandidate(priority, size, bufferWriterIdx, outputIdx, buffer));
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
    SmallVector<IntervalInfo> intervals;
    // store scheduled information
    if (scheduledOp.isImplicitWriteOp()) {
        // special case for a spill write with deallocation
        IntervalInfo interval;
        auto output = scheduledOp.spillBuffer_;
        // retrieve and store operation addresses
        interval.begin_ = checked_cast<size_t>(_scan.handler().getAddress(output));
        interval.end_ = interval.begin_ + checked_cast<size_t>(_scan.handler().getSize(output));
        interval.buffer_ = output;
        intervals.push_back(interval);
        // deallocate only after addresses stored
        _scan.handler().deallocate(output);
    } else {
        // retrieve interval information, operation can have multiple output buffers
        auto* bodyBlock = &_depsInfo.getExecuteOpAtIndex(scheduledOp.op_).body().front();
        for (auto& op : bodyBlock->getOperations()) {
            if (mlir::isa<IERT::LayerOpInterface>(op)) {
                auto outputs = mlir::dyn_cast<IERT::LayerOpInterface>(op).getOutputs();
                for (auto output : outputs) {
                    const auto type = output.getType().dyn_cast<mlir::MemRefType>();
                    if (type == nullptr || type.getMemorySpace() != _memSpace) {
                        continue;
                    }

                    // Find the root buffer for a given output as output of an operation
                    // doesn't have to point directly to result of memref.alloc (e.g. might
                    // be a result of SubView).
                    output = _aliasInfo.getRoot(output);

                    // in case of spill only allocate the spill buffer
                    if (!scheduledOp.isOriginalOp() && output != scheduledOp.spillBuffer_) {
                        continue;
                    }
                    IntervalInfo interval;
                    // store operation writing to the buffer
                    _opWritingToBuffer[output] = scheduledOp.op_;
                    // retrieve and store operation addresses
                    interval.begin_ = checked_cast<size_t>(_scan.handler().getAddress(output));
                    interval.end_ = interval.begin_ + checked_cast<size_t>(_scan.handler().getSize(output));
                    interval.buffer_ = output;
                    intervals.push_back(interval);
                }
            }
        }
    }
    // populate the struct fields
    ScheduledOpInfo scheduled;
    scheduled.op_ = scheduledOp.op_;
    scheduled.opType_ = scheduledOp.opType_;
    scheduled.time_ = scheduledOp.time_;
    scheduled.resourceInfo_ = intervals;
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
    scheduleAllPossibleReadyOpsAndUpdate(_readyComputeOps);
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
        bool pop_from_start_heap =
                start_top_ptr && (!completionTopPtr || (start_top_ptr->time_ < completionTopPtr->time_));

        if (pop_from_start_heap) {
            _log.trace("Popping from start time heap");
            // schedule first op in heap
            HeapElement firstOp = popFromStartTimeHeap();
            _currentTime = firstOp.time_;
            // add to output table
            populateScheduledOps(firstOp);
            // move to completion time heap
            pushToCompletionTimeHeap(HeapElement(firstOp.op_, firstOp.time_ + 1, firstOp.opType_));
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
                scheduleAllPossibleReadyOpsAndUpdate(_activeComputeOps);
                scheduleAllPossibleReadyOpsAndUpdate(_readyComputeOps);
            } while (!_completionTimeHeap.empty() && _startTimeHeap.empty());

            if (_startTimeHeap.empty()) {
                // unable to schedule an operation, perform spill
                _log.trace("Unable to schedule an operation, forcing dynamic spill");
                forceScheduleActiveOpEviction();
            }
        }
    }
}

llvm::SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo> FeasibleMemoryScheduler::generateSchedule() {
    init();
    // TODO: save schedule from _scheduledOps to file
    _log.trace("Generated Schedule");
    _log = _log.nest();
    for (const auto& op : _scheduledOps) {
        std::string resourceInfo = "<none>";
        if (op.hasActiveResource()) {
            resourceInfo = "";
            for (size_t resourceIdx = 0; resourceIdx < op.numOfResources(); resourceIdx++) {
                if (op.isActiveResource(resourceIdx)) {
                    resourceInfo += "resource = [" + std::to_string(op.beginResource(resourceIdx)) + " " +
                                    std::to_string(op.endResource(resourceIdx)) + "] size = " +
                                    std::to_string((op.endResource(resourceIdx) - op.beginResource(resourceIdx))) +
                                    ", ";
                }
            }
        }
        std::cout << "op = " << op.op_ << ", type = " << op.opTypeName().data() << ", time = " << op.time_ << ", "
                  << resourceInfo.data() << "\n";
        _log.trace("op = '{0}'\t type = '{1}'\t time = '{2}'\t '{3}'", op.op_, op.opTypeName(), op.time_, resourceInfo);
    }
    _log = _log.unnest();
    return _scheduledOps;
}
