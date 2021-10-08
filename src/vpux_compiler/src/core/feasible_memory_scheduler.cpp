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

//
// Constructor
//

FeasibleMemoryScheduler::FeasibleMemoryScheduler(mlir::Attribute& memSpace, MemLiveRangeInfo& liveRangeInfo,
                                                 AsyncDepsInfo& depsInfo,
                                                 LinearScan<mlir::Value, LinearScanHandler>& scan)
        : _memSpace(memSpace), _liveRangeInfo(liveRangeInfo), _depsInfo(depsInfo), _scan(scan) {
}

void FeasibleMemoryScheduler::pushToStartTimeHeap(const HeapElement& elem) {
    _startTimeHeap.push_back(elem);
    std::push_heap(_startTimeHeap.begin(), _startTimeHeap.end(), MinHeapOrdering());
}

FeasibleMemoryScheduler::HeapElement FeasibleMemoryScheduler::popFromStartTimeHeap() {
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
    std::pop_heap(_completionTimeHeap.begin(), _completionTimeHeap.end(), MinHeapOrdering());
    HeapElement elem = _completionTimeHeap.back();
    _completionTimeHeap.pop_back();
    return elem;
}

bool FeasibleMemoryScheduler::isDataOp(size_t opIdx) {
    StringRef dataType("DMA_NN");

    mlir::async::ExecuteOp op = _depsInfo.getExecuteOpAtIndex(opIdx);
    if (op->getAttr("IERT.executor").cast<mlir::StringAttr>().getValue().compare(dataType) != 0) {
        return false;
    }

    if (op.getOperands().size() == 0) {
        return true;
    }

    if (_outputOps.find(opIdx) != _outputOps.end()) {
        return false;
    }

    auto inOp = op.body().front().back().getOperand(0);
    bool isCMXMemSpace = true;
    if (auto memref = inOp.getType().dyn_cast<mlir::MemRefType>()) {
        if (memref.getMemorySpace()) {
            isCMXMemSpace = _memSpace == memref.getMemorySpace();
        }
    }

    return isCMXMemSpace;
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
    // free possible buffers
    const auto usedBufs = _liveRangeInfo.getUsedBuffers(op);
    for (auto val : usedBufs) {
        if (_liveRangeInfo.eraseUser(val, op) == 0) {
            _scan.handler().markAsDead(val);
        }
    }
    _scan.freeNonAlive();

    // decrement consumers
    for (auto dep : _depsInfo.getOpDeps(helement.op_)) {
        _opOutputTable.find(dep)->second.decrementConsumers();
    }

    auto _opOutput = _opOutputTable.find(helement.op_);
    if (_opOutput->second.consumed()) {
        _opOutput->second.changeStateToConsumed();
    }
}

bool FeasibleMemoryScheduler::isComputeOpWithSomeActiveInputs(size_t opIdx) {
    if (isDataOp(opIdx)) {
        return false;
    }
    // TODO: might need to port: active_resource_table
    auto opAllDeps = _depsInfo.getOpDeps(opIdx);
    auto opDeps = getSortedBuffers(opIdx);
    // number of buffers needing allocation smaller than number of buffers used by the op
    return opAllDeps.size() > opDeps.size();
}

vpux::AddressType FeasibleMemoryScheduler::calculateOpSize(size_t opIdx) {
    vpux::AddressType opSize = 0;
    auto execOp = _depsInfo.getExecuteOpAtIndex(opIdx);
    auto buffers = _liveRangeInfo.getUsedBuffers(execOp);
    for (auto& buf : buffers) {
        opSize += _scan.handler().getSize(buf);
    }
    return opSize;
}

void FeasibleMemoryScheduler::distributeReadyOps(llvm::ArrayRef<size_t> readyOps) {
    // populate ready lists depending on op type/state
    for (auto readyOp : readyOps) {
        vpux::AddressType opSize = calculateOpSize(readyOp);
        if (isDataOp(readyOp)) {
            // TODO: verify op not already in ready data ops
            _readyDataOps.insert(std::make_pair(readyOp, opSize));
            llvm::SmallVector<size_t> newReadyOps = reduceInDegreeOfAdjacentOperations(readyOp);
            distributeReadyOps(newReadyOps);
        } else if (isComputeOpWithSomeActiveInputs(readyOp)) {
            // TODO: verify op not already in active compute ops
            _activeComputeOps.insert(std::make_pair(readyOp, opSize));
        } else {
            // TODO: verify op not already in ready compute ops
            _readyComputeOps.insert(std::make_pair(readyOp, opSize));
        }
    }
}

void FeasibleMemoryScheduler::unscheduleAllCompletingOpsAtNextEarliestTime() {
    // retrieve the latest time
    const HeapElement* completionTopPtr = topElementGen(_completionTimeHeap);
    _currentTime = completionTopPtr->time_;

    llvm::SmallVector<HeapElement> unscheduledOps = popAllElementsAtThisTime(_currentTime);
    llvm::SmallVector<size_t> ready_ops = {};

    for (auto& op : unscheduledOps) {
        auto opIdx = op.op_;
        unscheduleOp(op);

        if (!isDataOp(opIdx) && op.isOriginalOp()) {
            // propagate through original compute ops
            auto newReadyOps = reduceInDegreeOfAdjacentOperations(opIdx);
            ready_ops.insert(ready_ops.end(), newReadyOps.begin(), newReadyOps.end());
        }
    }

    distributeReadyOps(ready_ops);
}

llvm::SmallVector<size_t> FeasibleMemoryScheduler::reduceInDegreeOfAdjacentOperations(size_t opIdx) {
    llvm::SmallVector<size_t> zeroIndegreeOps;
    // reduce indegree for consumers of ready data ops
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
    // populate ready data ops
    for (auto entry : _inDegreeTable) {
        if (entry.second == 0 && isDataOp(entry.first)) {
            vpux::AddressType opSize = calculateOpSize(entry.first);
            _readyDataOps.insert(std::make_pair(entry.first, opSize));
            // reduce indegree of op consumers
            reduceInDegreeOfAdjacentOperations(entry.first);
        }
    }
}

void FeasibleMemoryScheduler::getReadyComputeList() {
    // populate ready compute ops
    for (auto entry : _inDegreeTable) {
        if (entry.second == 0 && !isDataOp(entry.first)) {
            vpux::AddressType opSize = calculateOpSize(entry.first);
            _readyComputeOps.insert(std::make_pair(entry.first, opSize));
        }
    }
}

SmallVector<mlir::Value> FeasibleMemoryScheduler::getSortedBuffers(size_t opIdx) {
    auto op = _depsInfo.getExecuteOpAtIndex(opIdx);
    auto usedBuffs = _liveRangeInfo.getUsedBuffers(op);
    SmallVector<std::pair<vpux::AddressType, mlir::Value>> newBufs;
    for (auto val : usedBuffs) {
        const auto type = val.getType().cast<mlir::MemRefType>();
        if (type.getMemorySpace() != _memSpace || _scan.handler().isAlive(val)) {
            continue;
        }
        auto size = _scan.handler().getSize(val);
        newBufs.push_back(std::make_pair(size, val));
    }
    // sort based on size
    llvm::sort(newBufs.begin(), newBufs.end(),
               [](const std::pair<vpux::AddressType, mlir::Value>& val1,
                  const std::pair<vpux::AddressType, mlir::Value>& val2) {
                   return val1.first > val2.first;
               });
    SmallVector<mlir::Value> orderedBufs;
    for (auto pair : newBufs) {
        orderedBufs.push_back(pair.second);
    }
    return orderedBufs;
}

SmallVector<size_t> FeasibleMemoryScheduler::getNonEmptyOpDemandList(size_t opIdx) {
    SmallVector<size_t> demandList;
    for (auto dep : _depsInfo.getOpDeps(opIdx)) {
        if (_opOutputTable.find(dep) == _opOutputTable.end() || _opOutputTable[dep].spilled()) {
            demandList.push_back(dep);
        }
    }
    return demandList;
}

bool FeasibleMemoryScheduler::isReadyComputeOperationSchedulable(size_t opIdx) {
    // are resources available and can be allocated
    auto sortedBuffers = getSortedBuffers(opIdx);
    return _scan.canAlloc(sortedBuffers);
}

void FeasibleMemoryScheduler::scheduleInputOpForComputeOp(size_t inputIdx) {
    auto _opOutput = _opOutputTable.find(inputIdx);
    EOpType opType;
    if (_opOutput != _opOutputTable.end()) {
        (_opOutput->second).changeStateToActive();
        opType = EOpType::IMPLICIT_OP_READ;
        VPUX_THROW("Spill occured, spilling not yet implemented");
    } else {
        opType = EOpType::ORIGINAL_OP;
        _opOutputTable.insert(std::make_pair(inputIdx, OpOutputInfo(EOpState::ACTIVE, _outDegreeTable[inputIdx])));
    }
    pushToStartTimeHeap(HeapElement(inputIdx, _currentTime, opType));
}

void FeasibleMemoryScheduler::allocateSortedBuffers(ArrayRef<mlir::Value> sortedBuffers) {
    SmallVector<mlir::Value> buffersNeedingAllocation;
    for (auto val : sortedBuffers) {
        const auto type = val.getType().cast<mlir::MemRefType>();
        if (type.getMemorySpace() != _memSpace || _scan.handler().isAlive(val)) {
            continue;
        }
        _scan.handler().markAsAlive(val);
        buffersNeedingAllocation.push_back(val);
    }

    VPUX_THROW_UNLESS(_scan.alloc(buffersNeedingAllocation, /*allowSpills*/ false), "Failed LP allocation");
}

void FeasibleMemoryScheduler::scheduleComputeOp(size_t opIdx) {
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
        std::set<std::pair<size_t, vpux::AddressType>, SizeSort>& readyList) {
    SmallVector<std::pair<size_t, vpux::AddressType>> scheduledOps;
    // schedule ops that fit in CMX
    for (auto opIdx : readyList) {
        if (isReadyComputeOperationSchedulable(opIdx.first)) {
            scheduleComputeOp(opIdx.first);
            scheduledOps.push_back(opIdx);
        }
    }
    // update ready lists by removing scheduled ops
    for (auto scheduledOp : scheduledOps) {
        readyList.erase(scheduledOp);
    }
}

void FeasibleMemoryScheduler::populateScheduledOps(HeapElement& scheduledOp) {
    IntervalInfo interval;
    auto* bodyBlock = &_depsInfo.getExecuteOpAtIndex(scheduledOp.op_).body().front();
    for (auto& op : bodyBlock->getOperations()) {
        if (mlir::isa<mlir::ViewLikeOpInterface>(op) && mlir::isa<IERT::LayerOpInterface>(op)) {
            auto outputs = mlir::dyn_cast<IERT::LayerOpInterface>(op).getOutputs();
            for (auto output : outputs) {
                const auto type = output.getType().dyn_cast<mlir::MemRefType>();
                if (type == nullptr || type.getMemorySpace() != _memSpace) {
                    continue;
                }
                interval.begin_ = checked_cast<size_t>(_scan.handler().getAddress(output));
                interval.end_ = interval.begin_ + checked_cast<size_t>(_scan.handler().getSize(output));
            }
        }
    }
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
        // choose the minimum time
        const HeapElement* start_top_ptr = topElementGen(_startTimeHeap);
        const HeapElement* completionTopPtr = topElementGen(_completionTimeHeap);

        bool pop_from_start_heap =
                start_top_ptr && (!completionTopPtr || (start_top_ptr->time_ < completionTopPtr->time_));

        if (pop_from_start_heap) {
            // schedule first op in heap
            HeapElement firstOp = popFromStartTimeHeap();
            _currentTime = firstOp.time_;
            _timeBuckets[firstOp.time_].push_back(firstOp.op_);
            // add to output table
            populateScheduledOps(firstOp);
            // move to completion time heap
            pushToCompletionTimeHeap(HeapElement(firstOp.op_, firstOp.time_ + 1, firstOp.opType_));
            // decrease outputs ops if output op scheduled
            if (_outputOps.find(firstOp.op_) != _outputOps.end()) {
                _outputOps.erase(firstOp.op_);
            }
        } else {
            do {
                // unschedule operations, and propagate through the graph by creating new ready ops
                unscheduleAllCompletingOpsAtNextEarliestTime();
                scheduleAllPossibleReadyOpsAndUpdate(_activeComputeOps);
                scheduleAllPossibleReadyOpsAndUpdate(_readyComputeOps);
            } while (!_completionTimeHeap.empty() && _startTimeHeap.empty());

            if (_startTimeHeap.empty()) {
                // unable to schedule an operation, perform spill
                VPUX_THROW("Spill required, dynamic spilling not yet implemented");
            }
        }
    }
}

llvm::SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo> FeasibleMemoryScheduler::generateSchedule() {
    init();
    // TODO: save schedule to file
    std::cout << "\n #### INITIAL SCHEDULE ####\n" << std::endl;
    for (auto op : _scheduledOps) {
        std::string resourceInfo = "<none>";
        if (op.hasActiveResource()) {
            resourceInfo = "resource = [" + std::to_string(op.beginResource()) + " " +
                           std::to_string(op.endResource()) +
                           "] size = " + std::to_string((op.endResource() - op.beginResource()));
        }
        std::cout << "op = " << op.op_ << "\ttype = " << op.opTypeName() << "\ttime = " << op.time_ << "\t"
                  << resourceInfo << std::endl;
    }
    return _scheduledOps;
}
