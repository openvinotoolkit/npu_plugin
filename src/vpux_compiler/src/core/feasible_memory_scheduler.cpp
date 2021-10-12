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

// This class will try to porduce a feasible memory schedule based on the dependency map provided from
// AsyncDepsInfo and use the LinearScan class to allocate the resources.
// Data and Compute ops, where Data ops are operations moving data to CMX are distinguished in order to
// follow the scheduling of Compute ops along with their dependencies (Data ops). This optimizes CMX usage,
// and allows for feasible CMX schedule to be generated.
// The graph is iterated topologically based on the dependencies from input to output(s).
// In init() the input will be considered as a compute operation, this will be the first ready compute operation.
// In nextSchedulableOp() there are two possible scenarios:
// 1. Scheudling the next earlies operation from the start time heap, and adding it to the op output table.
// 2. Unscheduling operations: freeing CMX space and updating dependencies, creating new ready
//      operations which will be allocated at the next time slot.

FeasibleMemoryScheduler::FeasibleMemoryScheduler(mlir::Attribute& memSpace, MemLiveRangeInfo& liveRangeInfo,
                                                 AsyncDepsInfo& depsInfo, Logger& log,
                                                 LinearScan<mlir::Value, LinearScanHandler>& scan)
        : _log(log), _memSpace(memSpace), _liveRangeInfo(liveRangeInfo), _depsInfo(depsInfo), _scan(scan) {
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
    // considered compute operations. This disthinguishment is neeeded to balance
    // CMX memory space and not to fill CMX space with only data operations resulting
    // in not being able to fit the compute operation. Data operations will only be
    // scheduled when needed by the compute operation so that the CMX space can be
    // freed as soon as possible.
    auto op = _depsInfo.getExecuteOpAtIndex(opIdx);
    if (op->hasAttr(IERT::IERTDialect::getExecutorAttrName())) {
        uint32_t numUnits;
        const auto executor = IERT::IERTDialect::getExecutor(op, numUnits);
        if (executor == VPUIP::DMAEngineAttr::get(op->getContext(), VPUIP::DMAEngine::DMA_NN)) {
            if (_outputOps.find(opIdx) != _outputOps.end()) {
                return false;
            }
            auto* bodyBlock = &op.body().front();
            if (op.getOperands().empty()) {
                for (auto& op : bodyBlock->getOperations()) {
                    for (const auto& operand : op.getOperands()) {
                        if (operand.getDefiningOp() == nullptr) {
                            // operation using function input
                            // input considered to be a compute operation
                            return false;
                        }
                    }
                }
            }
            for (auto& op : bodyBlock->getOperations()) {
                if (mlir::isa<IERT::CopyOp>(op)) {
                    if (auto copyOp = mlir::dyn_cast<IERT::CopyOp>(op)) {
                        // DMA to NN_CMX
                        return _memSpace == copyOp.output().getType().dyn_cast<mlir::MemRefType>().getMemorySpace();
                    }
                }
            }
        }
    }
    return false;
}

FeasibleMemoryScheduler::HeapElement const* FeasibleMemoryScheduler::topElementGen(
        const llvm::SmallVector<HeapElement>& heap) const {
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

void FeasibleMemoryScheduler::unscheduleOp(const HeapElement& helement) {
    auto op = _depsInfo.getExecuteOpAtIndex(helement.op_);
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

    // decrement consumers of the op
    for (auto dep : _depsInfo.getOpDeps(helement.op_)) {
        _opOutputTable.find(dep)->second.decrementConsumers();
    }

    auto _opOutput = _opOutputTable.find(helement.op_);
    if (_opOutput->second.consumed()) {
        _opOutput->second.changeStateToConsumed();
    }
}

bool FeasibleMemoryScheduler::isComputeOpWithSomeActiveInputs(operationIdxType opIdx) {
    if (isDataOp(opIdx)) {
        return false;
    }
    // TODO-EISW-21295: might need to port: active_resource_table
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
        if (mlir::isa<mlir::ViewLikeOpInterface>(op) && mlir::isa<IERT::LayerOpInterface>(op)) {
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
        if (isDataOp(readyOp)) {
            // TODO-EISW-21295: verify op not already in ready data ops
            _readyDataOps.insert(std::make_pair(readyOp, opSize));
            _log.trace("Add to ready data ops '{0}'", readyOp);
            llvm::SmallVector<operationIdxType> newReadyOps = reduceInDegreeOfAdjacentOperations(readyOp);
            distributeReadyOps(newReadyOps);
        } else if (isComputeOpWithSomeActiveInputs(readyOp)) {
            // TODO-EISW-21295: verify op not already in active compute ops
            _activeComputeOps.insert(std::make_pair(readyOp, opSize));
            _log.trace("Add to active compute ops '{0}'", readyOp);
        } else {
            // TODO-EISW-21295: verify op not already in ready compute ops
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
    llvm::SmallVector<operationIdxType> zeroIndegreeOps;
    // reduce indegree (number of incoming edges) for consumers of ready data ops
    for (auto consumer : _depsInfo.getConsumerOps(opIdx)) {
        if (_inDegreeTable[consumer] < 2) {
            zeroIndegreeOps.push_back(consumer);
            _inDegreeTable.erase(consumer);
        } else {
            VPUX_THROW_UNLESS(_inDegreeTable[consumer] > 0, "Invalid indegree");
            _inDegreeTable[consumer]--;
        }
    }
    return zeroIndegreeOps;
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

SmallVector<operationIdxType> FeasibleMemoryScheduler::getNonEmptyOpDemandList(operationIdxType opIdx) {
    // return all buffers of an op that require allocation
    SmallVector<operationIdxType> demandList;
    for (auto& dep : _depsInfo.getOpDeps(opIdx)) {
        if (_opOutputTable.find(dep) == _opOutputTable.end() || _opOutputTable[dep].spilled()) {
            demandList.push_back(dep);
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
    auto _opOutput = _opOutputTable.find(inputIdx);
    EOpType opType;
    if (_opOutput != _opOutputTable.end()) {
        (_opOutput->second).changeStateToActive();
        opType = EOpType::IMPLICIT_OP_READ;
    } else {
        opType = EOpType::ORIGINAL_OP;
        _opOutputTable.insert(std::make_pair(inputIdx, OpOutputInfo(EOpState::ACTIVE, _outDegreeTable[inputIdx])));
    }
    _log.nest().trace("Scheduling input for compute op:'{0}'", inputIdx);
    pushToStartTimeHeap(HeapElement(inputIdx, _currentTime, opType));
}

void FeasibleMemoryScheduler::allocateSortedBuffers(ArrayRef<mlir::Value> sortedBuffers) {
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
    }
    _log.nest().trace("Allocate memory for the alive buffers");
    // allocate buffers using LinearScan
    VPUX_THROW_UNLESS(_scan.alloc(buffersNeedingAllocation, /*allowSpills*/ false),
                      "Failed to statically allocate '{0}' memory", _memSpace);
}

void FeasibleMemoryScheduler::scheduleComputeOp(operationIdxType opIdx) {
    // Step 1: add to output result table
    _opOutputTable.insert(std::make_pair(opIdx, OpOutputInfo(EOpState::ACTIVE, _outDegreeTable[opIdx])));

    // Step 2: assign resources simultaneously
    auto demandList = getNonEmptyOpDemandList(opIdx);
    size_t maxInputDelay = 0;
    for (auto demand : demandList) {
        scheduleInputOpForComputeOp(demand);
        maxInputDelay = 1;
    }
    auto sortedBuffers = getSortedBuffers(opIdx);
    allocateSortedBuffers(sortedBuffers);

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

void FeasibleMemoryScheduler::evictActiveOp(operationIdxType opIdx, mlir::Value* buffer) {
    auto opOutput = _opOutputTable.find(opIdx);
    assert(opOutput != _opOutputTable.end() && opOutput->second.active());
    opOutput->second.changeStateToSpilled();
    auto buf = *buffer;

    _scan.handler().markAsDead(buf);
    _scan.freeNonAlive();
}

void FeasibleMemoryScheduler::forceScheduleActiveOpEviction() {
    std::cout << "choose_active_operation_for_eviction" << std::endl;
    auto smallestAlive = _scan.handler().getSmallestBufferAlive();
    VPUX_THROW_UNLESS(smallestAlive != nullptr, "Failed, nothing to spill");

    std::cout << "smallest buffer alive size " << _scan.handler().getSize(*smallestAlive) << std::endl;
    (*smallestAlive).dump();

    // TODO: assert only 1 user, or always choose last user ?
    mlir::Operation* parentOp = nullptr;
    for (auto user : (*smallestAlive).getDefiningOp()->getUsers()) {
        user->dump();
        parentOp = user->getParentOp();
    }
    auto execOp = mlir::cast<mlir::async::ExecuteOp>(*parentOp);

    // return;

    size_t candidateIdx = _depsInfo.getIndex(execOp);
    std::cout << "Candidate for spill: " << candidateIdx << std::endl;
    evictActiveOp(candidateIdx, smallestAlive);
    _opOutputTable[candidateIdx].changeStateToSpilled();

    // TODO: update _activeComputeOps some may no longer be active
    pushToStartTimeHeap(HeapElement(candidateIdx, _currentTime, EOpType::IMPLICIT_OP_WRITE));
}

void FeasibleMemoryScheduler::populateScheduledOps(HeapElement& scheduledOp) {
    IntervalInfo interval;
    // retrieve interval information
    auto* bodyBlock = &_depsInfo.getExecuteOpAtIndex(scheduledOp.op_).body().front();
    for (auto& op : bodyBlock->getOperations()) {
        if (mlir::isa<mlir::ViewLikeOpInterface>(op) && mlir::isa<IERT::LayerOpInterface>(op)) {
            auto outputs = mlir::dyn_cast<IERT::LayerOpInterface>(op).getOutputs();
            for (const auto& output : outputs) {
                const auto type = output.getType().dyn_cast<mlir::MemRefType>();
                if (type == nullptr || type.getMemorySpace() != _memSpace) {
                    continue;
                }
                interval.begin_ = checked_cast<size_t>(_scan.handler().getAddress(output));
                interval.end_ = interval.begin_ + checked_cast<size_t>(_scan.handler().getSize(output));
                if (scheduledOp.isImplicitWriteOp()) {
                    _scan.handler().deallocate(output);
                }
            }
        }
    }
    // populate the struct fields
    ScheduledOpInfo scheduled;
    scheduled.op_ = scheduledOp.op_;
    scheduled.opType_ = scheduledOp.opType_;
    scheduled.time_ = scheduledOp.time_;
    scheduled.resourceInfo_ = interval;
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
                std::cout << "unable to schedule an operation, perform spill" << std::endl;
                std::cout << "total CMX free size: " << _scan.totalFreeSize() << std::endl;
                std::cout << "max CMX free size: " << _scan.maxFreeSize() << std::endl;
                std::cout << "number of live ranges: " << _scan.liveRanges().size() << std::endl;
                std::cout << "number of gaps: " << _scan.gaps().size() << std::endl;
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
    for (auto op : _scheduledOps) {
        std::string resourceInfo = "<none>";
        if (op.hasActiveResource()) {
            resourceInfo = "resource = [" + std::to_string(op.beginResource()) + " " +
                           std::to_string(op.endResource()) +
                           "] size = " + std::to_string((op.endResource() - op.beginResource()));
        }
        _log.trace("op = '{0}'\t type = '{1}'\t time = '{2}'\t '{3}'", op.op_, op.opTypeName(), op.time_, resourceInfo);
    }
    _log = _log.unnest();
    return _scheduledOps;
}
