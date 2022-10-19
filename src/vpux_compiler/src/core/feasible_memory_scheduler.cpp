//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/feasible_memory_scheduler.hpp"

#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
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
// In init() ready lists will be populated using operations without dependencies.
// In schedulingLoop() there are two possible scenarios:
// 1. Scheduling the next earliest operation from the start cycle heap, and adding it to the op output table.
// 2. Unscheduling operations: freeing CMX space and updating dependencies, creating new ready
//      operations which will be allocated at the next availible cycle.

FeasibleMemoryScheduler::FeasibleMemoryScheduler(VPU::MemoryKind memKind, MemLiveRangeInfo& liveRangeInfo,
                                                 AsyncDepsInfo& depsInfo, AliasesInfo& aliasInfo, Logger log,
                                                 LinearScan<mlir::Value, LinearScanHandler>& scan, VPU::ArchKind arch,
                                                 std::shared_ptr<VPUNN::VPUCostModel> costModel,
                                                 int64_t nceClusterCount)
        : _log(log),
          _memKind(memKind),
          _liveRangeInfo(liveRangeInfo),
          _depsInfo(depsInfo),
          _aliasInfo(aliasInfo),
          _scan(scan),
          _archKind(arch),
          _costModel(costModel),
          _nceClusterCount(nceClusterCount) {
}

void FeasibleMemoryScheduler::pushToCycleBeginHeap(const HeapElement& elem) {
    _cycleBeginHeap.push_back(elem);
    std::push_heap(_cycleBeginHeap.begin(), _cycleBeginHeap.end(), MinHeapOrdering());
}

FeasibleMemoryScheduler::HeapElement FeasibleMemoryScheduler::popFromCycleBeginHeap() {
    VPUX_THROW_UNLESS(!_cycleBeginHeap.empty(), "Tried to pop from empty _cycleBeginHeap");
    std::pop_heap(_cycleBeginHeap.begin(), _cycleBeginHeap.end(), MinHeapOrdering());
    HeapElement elem = _cycleBeginHeap.back();
    _cycleBeginHeap.pop_back();
    return elem;
}

void FeasibleMemoryScheduler::pushToCycleEndHeap(const HeapElement& elem) {
    _cycleEndHeap.push_back(elem);
    std::push_heap(_cycleEndHeap.begin(), _cycleEndHeap.end(), MinHeapOrdering());
}

FeasibleMemoryScheduler::HeapElement FeasibleMemoryScheduler::popFromCycleEndHeap() {
    VPUX_THROW_UNLESS(!_cycleEndHeap.empty(), "Tried to pop from empty _cycleEndHeap");
    std::pop_heap(_cycleEndHeap.begin(), _cycleEndHeap.end(), MinHeapOrdering());
    HeapElement elem = _cycleEndHeap.back();
    _cycleEndHeap.pop_back();
    return elem;
}

VPU::ExecutorKind FeasibleMemoryScheduler::getExecutorType(operationIdxType opIdx) {
    if (_opOutputTable.find(opIdx) != _opOutputTable.end() && _opOutputTable[opIdx].spilled()) {
        // spilled operation using DMAs for relocation
        return VPU::ExecutorKind::DMA_NN;
    }
    auto execOp = _depsInfo.getExecuteOpAtIndex(opIdx);
    if (execOp->hasAttr(VPUIP::VPUIPDialect::getExecutorAttrName())) {
        const auto executor = VPUIP::VPUIPDialect::getExecutor(execOp);
        if (executor.getLeafNameAttr() ==
            VPU::ExecutorKindAttr::get(execOp->getContext(), VPU::ExecutorKind::SHAVE_UPA)) {
            return VPU::ExecutorKind::SHAVE_UPA;
        } else if (executor.getLeafNameAttr() ==
                   VPU::ExecutorKindAttr::get(execOp->getContext(), VPU::ExecutorKind::DMA_NN)) {
            return VPU::ExecutorKind::DMA_NN;
        } else if (executor.getLeafNameAttr() ==
                   VPU::ExecutorKindAttr::get(execOp->getContext(), VPU::ExecutorKind::NCE)) {
            return VPU::ExecutorKind::NCE;
        }
    }
    // for now treat all other executors as NCE - same as previous implementation
    return VPU::ExecutorKind::NCE;
}

size_t FeasibleMemoryScheduler::getCurrentCycle(operationIdxType opIdx) {
    auto executor = getExecutorType(opIdx);
    auto earliestBeginCycle = _executorPipelines[executor];
    // check if operation cycle begin delayed by dependencies
    for (auto& dep : _depsInfo.getOpDeps(opIdx)) {
        earliestBeginCycle = std::max(earliestBeginCycle, getOperationEndCycle(dep, earliestBeginCycle));
    }
    return earliestBeginCycle;
}

void FeasibleMemoryScheduler::updateCurrentCycleForExecutor(operationIdxType opIdx, size_t nextAvailableCycle,
                                                            bool spilled) {
    if (spilled) {
        _executorPipelines[VPU::ExecutorKind::DMA_NN] = nextAvailableCycle;
    } else {
        _executorPipelines[getExecutorType(opIdx)] = nextAvailableCycle;
    }
}

size_t FeasibleMemoryScheduler::spilledOperationCycleCost(operationIdxType opIdx) {
    if (_spillOperationCycleCost.find(opIdx) != _spillOperationCycleCost.end()) {
        // reuse calculated cycles
        return _spillOperationCycleCost[opIdx];
    }
    // get cost of inner spill
    auto execOp = _depsInfo.getExecuteOpAtIndex(opIdx);
    mlir::Operation* currentOp = nullptr;
    auto* bodyBlock = &execOp.body().front();
    for (auto& innerOp : bodyBlock->getOperations()) {
        if (auto nceClustOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(innerOp)) {
            if (nceClustOp.getInnerTaskOp()->hasAttr(DPUCost) ||
                mlir::isa<VPUIP::CopyOp>(nceClustOp.getInnerTaskOp())) {
                currentOp = nceClustOp.getInnerTaskOp();
            }
        } else if (innerOp.hasAttr(DPUCost) || mlir::isa<VPUIP::CopyOp>(innerOp)) {
            currentOp = (&innerOp);
        }
    }
    _spillOperationCycleCost[opIdx] =
            getDMACost(currentOp->getResult(0), currentOp->getResult(0), _archKind, _costModel);
    return _spillOperationCycleCost[opIdx];
}

size_t FeasibleMemoryScheduler::operationCycleCost(operationIdxType opIdx, bool spilled) {
    auto execOp = _depsInfo.getExecuteOpAtIndex(opIdx);
    if (!execOp->hasAttr(cycleCostAttrName)) {
        // operations without cycle cost will have cycle cost = 1
        _log.warning("async.exec {0} has no cycle cost attribute {1}", execOp->getLoc(), cycleCostAttrName);
        return 1;
    }

    if (spilled || (_opOutputTable.find(opIdx) != _opOutputTable.end() && _opOutputTable[opIdx].spilled())) {
        // spilled operation using DMAs for relocation
        return spilledOperationCycleCost(opIdx);
    }

    return checked_cast<size_t>(execOp->getAttr(cycleCostAttrName).cast<mlir::IntegerAttr>().getValue().getSExtValue());
}

bool FeasibleMemoryScheduler::isDataOp(operationIdxType opIdx) {
    // Operations moving data to CMX are considered data ops. All others are
    // considered compute operations. This distinguishment is needed to balance
    // CMX memory space and not to fill CMX space with only data operations resulting
    // in not being able to fit the compute operation. Data operations will only be
    // scheduled when needed by the compute operation so that the CMX space can be
    // freed as soon as possible.
    auto op = _depsInfo.getExecuteOpAtIndex(opIdx);

    if (!op->hasAttr(VPUIP::VPUIPDialect::getExecutorAttrName())) {
        return false;
    }

    const auto executor = VPUIP::VPUIPDialect::getExecutor(op);
    if (executor.getLeafNameAttr() != VPU::ExecutorKindAttr::get(op->getContext(), VPU::ExecutorKind::DMA_NN)) {
        return false;
    }

    if (_outputOps.find(opIdx) != _outputOps.end()) {
        return false;
    }

    auto* bodyBlock = &op.body().front();
    for (auto& innerOp : bodyBlock->getOperations()) {
        VPUIP::CopyOp copyOp;
        // CopyOp can be placed directly in async exec op or wrapped with NCEClusterTiling
        if (auto nceClustOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(innerOp)) {
            copyOp = mlir::dyn_cast<VPUIP::CopyOp>(nceClustOp.getInnerTaskOp());
        } else {
            copyOp = mlir::dyn_cast<VPUIP::CopyOp>(innerOp);
        }

        if (copyOp) {
            // DMA from DDR to NN_CMX
            auto srcMemSpace = copyOp.input().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
            auto dstMemSpace = copyOp.output().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
            return (_memKind == dstMemSpace && _memKind != srcMemSpace);
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
        VPUIP::CopyOp copyOp;
        // CopyOp can be placed directly in async exec op or wrapped with NCEClusterTiling
        if (auto nceClustOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(innerOp)) {
            copyOp = mlir::dyn_cast<VPUIP::CopyOp>(nceClustOp.getInnerTaskOp());
        } else {
            copyOp = mlir::dyn_cast<VPUIP::CopyOp>(innerOp);
        }

        if (copyOp) {
            auto dstMemSpace = copyOp.output().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
            return _memKind != dstMemSpace;
        }
    }

    return false;
}

FeasibleMemoryScheduler::HeapElement const* FeasibleMemoryScheduler::topElementGen(ArrayRef<HeapElement> heap) const {
    return heap.empty() ? nullptr : &(heap.front());
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
    if (!hElemet.isSpillWriteOp()) {
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

void FeasibleMemoryScheduler::distributeReadyOps(llvm::ArrayRef<operationIdxType> readyOps) {
    // populate ready lists depending on op type/state
    _log.trace("Distribute new ready ops");
    _log = _log.nest();
    for (auto& readyOpIdx : readyOps) {
        if (isDataOp(readyOpIdx)) {
            VPUX_THROW_UNLESS(_readyDataOps.find(readyOpIdx) == _readyDataOps.end(),
                              "Operation already in the ready data list '{0}'", readyOpIdx);
            _readyDataOps.insert(readyOpIdx);
            _log.trace("Add to ready data ops '{0}'", readyOpIdx);
            const auto newReadyOps = reduceInDegreeOfAdjacentOperations(readyOpIdx);
            distributeReadyOps(newReadyOps);
        } else if (isNonComputeChainOp(readyOpIdx)) {
            VPUX_THROW_UNLESS(_nonComputeChainOps.find(readyOpIdx) == _nonComputeChainOps.end(),
                              "Operation already in non compute chain op list '{0}'", readyOpIdx);
            _nonComputeChainOps.insert(readyOpIdx);
            _log.trace("Non compute chain op ready '{0}'", readyOpIdx);
        } else {
            VPUX_THROW_UNLESS(_readyComputeOps.find(readyOpIdx) == _readyComputeOps.end(),
                              "Operation already in ready compute list '{0}'", readyOpIdx);
            _readyComputeOps.insert(readyOpIdx);
            _log.trace("Add to ready compute ops '{0}'", readyOpIdx);
        }
    }
    _log = _log.unnest();
}

void FeasibleMemoryScheduler::unscheduleAllCompletingOps() {
    // unschedule all operations from cycle end heap
    SmallVector<operationIdxType> readyOps = {};

    _log = _log.nest();
    for (auto& op : _cycleEndHeap) {
        auto opIdx = op.op_;
        _log.trace("Unscheduling '{0}'", opIdx);
        unscheduleOp(op);
        if (!isDataOp(opIdx) && op.isOriginalOp()) {
            // propagate through original compute ops, generate new ready ops
            auto newReadyOps = reduceInDegreeOfAdjacentOperations(opIdx);
            _log.nest().trace("Reduce consumer indegree");
            readyOps.insert(readyOps.end(), newReadyOps.begin(), newReadyOps.end());
        }
    }
    _log = _log.unnest();

    _cycleEndHeap.clear();
    distributeReadyOps(readyOps);
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

void FeasibleMemoryScheduler::initializeReadyLists() {
    // populate ready lists with operations without dependencies
    SmallVector<operationIdxType> operationsWithNoDependencies;

    for (auto& entry : _inDegreeTable) {
        if (entry.second == 0) {
            operationsWithNoDependencies.push_back(entry.first);
        }
    }

    distributeReadyOps(operationsWithNoDependencies);
}

SmallVector<mlir::Value> FeasibleMemoryScheduler::sortUsedBuffers(mlir::DenseSet<mlir::Value>& operationBuffers) {
    // retrieve size of buffers
    SmallVector<BufferOrder> bufferVector;
    // order buffers based on usage type
    for (auto& val : operationBuffers) {
        auto opSize = _scan.handler().getSize(val);
        size_t opLevel = std::numeric_limits<size_t>::min();
        bool allocateFirst = false;
        if (_bufferLevels.find(val) != _bufferLevels.end()) {
            opLevel = _bufferLevels[val];
        }
        for (auto user : val.getUsers()) {
            if (user->hasAttr("exceedingNNCMX")) {
                // allocate exceeding buffers first to not exceed NNCMX
                _log.trace("Re-ordering exceeding NNCMX buffer: '{0}'", val);
                allocateFirst = true;
            }
        }
        bufferVector.push_back(BufferOrder(val, opSize, opLevel, allocateFirst));
    }
    // sort based on buffer qualities
    llvm::sort(bufferVector.begin(), bufferVector.end(), [](const BufferOrder& val1, const BufferOrder& val2) {
        // first special buffers
        if (val1.highAllocationPriority != val2.highAllocationPriority) {
            if (val1.highAllocationPriority) {
                return true;
            } else if (val2.highAllocationPriority) {
                return false;
            }
        }

        // second level of operation
        if (val1.level != val2.level) {
            return val1.level < val2.level;
        }

        // third op size
        if (val1.size != val2.size) {
            return val1.size > val2.size;
        }

        // finally position in IR
        const auto parentOp = val1.buffer.getDefiningOp();
        VPUX_THROW_UNLESS(parentOp != nullptr, "Block arguments are not supported");
        return parentOp->isBeforeInBlock(val2.buffer.getDefiningOp());
    });

    // repopulate only with buffers
    SmallVector<mlir::Value> orderedBufs;
    for (auto& buff : bufferVector) {
        orderedBufs.push_back(buff.buffer);
    }
    return orderedBufs;
}

bool FeasibleMemoryScheduler::hasBuffersInTargetMemoryKind(operationIdxType opIdx) {
    // check if operation has buffers in target memory kind
    auto op = _depsInfo.getExecuteOpAtIndex(opIdx);
    auto usedBuffs = _liveRangeInfo.getUsedBuffers(op);

    for (auto& buffer : usedBuffs) {
        auto rootBuffers = _aliasInfo.getRoots(buffer);
        VPUX_THROW_UNLESS(rootBuffers.size() == 1, "Value '{0}' expected to have only one root. Got {1}", buffer,
                          rootBuffers.size());
        const auto rootBuffer = *rootBuffers.begin();
        const auto type = rootBuffer.getType().cast<vpux::NDTypeInterface>();
        if (type.getMemoryKind() != _memKind) {
            continue;
        }
        return true;
    }
    return false;
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
        if (type.getMemoryKind() != _memKind || _scan.handler().isAlive(rootBuffer)) {
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
    // preserve order of NCE
    size_t levelOfBuffers = 0;
    if (!_prefetchSchedule.empty()) {
        if (_prefetchSchedule.front().computeOpIdx != opIdx && getExecutorType(opIdx) == VPU::ExecutorKind::NCE) {
            return false;
        } else {
            levelOfBuffers = _prefetchSchedule.front().computeOpLevel;
        }
    }
    // retrieve op demand list - input ops
    auto usedBuffers = getNonAliveBuffersUsedByOperation(opIdx);
    auto demandList = getNonEmptyOpDemandList(opIdx, usedBuffers);
    mlir::DenseSet<mlir::Value> buffersNeedingAllocation;

    // retrieve operation's buffers that need allocation
    for (auto val : getNonAliveBuffersUsedByOperation(opIdx)) {
        buffersNeedingAllocation.insert(val);
        _bufferLevels[val] = levelOfBuffers;
    }

    // retrieve operation input's buffers
    for (auto inputIdx : demandList) {
        for (auto val : getNonAliveBuffersUsedByOperation(inputIdx)) {
            buffersNeedingAllocation.insert(val);
            _bufferLevels[val] = levelOfBuffers;
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

size_t FeasibleMemoryScheduler::scheduleInputOpForComputeOp(operationIdxType inputIdx) {
    // schedule the dependency - Data op
    auto scheduleCycle = getCurrentCycle(inputIdx);
    _log.nest().trace("Scheduling input for compute op:'{0}' at cycle {1}", inputIdx, scheduleCycle);
    _opOutputTable.insert(std::make_pair(inputIdx, OpOutputInfo(EOpState::ACTIVE, _outDegreeTable[inputIdx])));
    // update current cycle directly
    auto nextAvailibleCycle = scheduleCycle + operationCycleCost(inputIdx);
    updateCurrentCycleForExecutor(inputIdx, nextAvailibleCycle);
    pushToCycleBeginHeap(HeapElement(inputIdx, scheduleCycle, nextAvailibleCycle, EOpType::ORIGINAL_OP));
    return nextAvailibleCycle;
}

size_t FeasibleMemoryScheduler::schedulePrefetchOp(operationIdxType inputIdx) {
    // mark buffers as alive
    for (auto& buff : getNonAliveBuffersUsedByOperation(inputIdx)) {
        _scan.handler().markAsAlive(buff);
    }
    // schedule the prefetch op
    auto scheduleCycle = getCurrentCycle(inputIdx);
    _log.nest().trace("Scheduling prefetched data op:'{0}' at cycle {1}", inputIdx, scheduleCycle);
    _opOutputTable.insert(std::make_pair(inputIdx, OpOutputInfo(EOpState::ACTIVE, _outDegreeTable[inputIdx])));
    // update current cycle directly
    auto nextAvailibleCycle = scheduleCycle + operationCycleCost(inputIdx);
    updateCurrentCycleForExecutor(inputIdx, nextAvailibleCycle);
    pushToCycleBeginHeap(HeapElement(inputIdx, scheduleCycle, nextAvailibleCycle, EOpType::ORIGINAL_PREFETCHED_OP));
    return nextAvailibleCycle;
}

size_t FeasibleMemoryScheduler::scheduleSpilledOpBuffer(operationIdxType inputIdx, mlir::Value* buffer) {
    // schedule the spilled dependency
    auto scheduleCycle = getCurrentCycle(inputIdx);
    _log.nest().trace("Scheduling spilled op:'{0}' at cycle {1}", inputIdx, scheduleCycle);
    // also store the buffer spilled
    auto spilledReadBuffer = *buffer;
    // update current cycle directly
    auto nextAvailibleCycle = scheduleCycle + operationCycleCost(inputIdx, true);
    updateCurrentCycleForExecutor(inputIdx, nextAvailibleCycle);
    pushToCycleBeginHeap(HeapElement(inputIdx, scheduleCycle, nextAvailibleCycle, EOpType::IMPLICIT_SPILL_READ_OP,
                                     spilledReadBuffer));
    // update output table after cycles assigned
    auto _opOutput = _opOutputTable.find(inputIdx);
    if (!_opOutput->second.spillIdx_.empty()) {
        _opOutput->second.spillIdx_.erase(getOpBufferOutputIdx(inputIdx, *buffer));
    }
    (_opOutput->second).changeStateToActive();
    return nextAvailibleCycle;
}

void FeasibleMemoryScheduler::allocatePrefetchOps(operationIdxType opIdx, size_t earliestComputeBeginCycle,
                                                  mlir::DenseSet<mlir::Value>& buffersNeedingAllocation) {
    // prefetch DMA operations to current DPU
    auto computeEndCycle = earliestComputeBeginCycle + operationCycleCost(opIdx);
    // check if any prefetched operations are linked to this op
    prefetchSet notAllocatedPrefetchOps;
    SmallVector<EvictionCandidate> notAllocatedSpilledOps;
    // limit number of prefetch to barrier limitations
    // count buffers used by the NCE, extra dependency from linearization NCE->NCE, and count from 0
    size_t barrierLimit =
            checked_cast<size_t>(_barrierPerCluster * _nceClusterCount) - buffersNeedingAllocation.size() - 2;
    size_t prefetchCount = 0;
    auto prefetchFront = _prefetchSchedule.begin();
    if (prefetchFront != _prefetchSchedule.end()) {
        if (prefetchFront->computeOpIdx == opIdx) {
            if (!prefetchFront->dataOps.empty()) {
                // allow space for all required buffers
                auto buffersPlusPrefetch = buffersNeedingAllocation;
                for (auto prefetchDMA : prefetchFront->dataOps) {
                    _dataOpLevels[prefetchDMA.opIdx_] = prefetchDMA.level_;

                    // if operation is ready
                    auto inDegree = _inDegreeTable.find(prefetchDMA.opIdx_);
                    if (inDegree == _inDegreeTable.end() || inDegree->second == 0) {
                        // try to prefetch future DMAs
                        auto buffersPlusCurrentPrefetch = buffersNeedingAllocation;

                        if (prefetchDMA.buffer_ == nullptr) {
                            // first attempt to prefetch DMA
                            _log.nest(2).trace("Try to prefetch data '{0}' with level '{1}'", prefetchDMA.opIdx_,
                                               prefetchDMA.level_);
                            for (auto& buff : getNonAliveBuffersUsedByOperation(prefetchDMA.opIdx_)) {
                                buffersPlusCurrentPrefetch.insert(buff);
                                buffersPlusPrefetch.insert(buff);
                                if (_bufferLevels.find(buff) != _bufferLevels.end()) {
                                    // do not overwrite buffers with level
                                    continue;
                                }
                                _bufferLevels[buff] = prefetchDMA.level_;
                            }
                        } else {
                            if (_scan.handler().isAlive(prefetchDMA.buffer_)) {
                                continue;
                            }
                            // attempt to prefetch spilled DMA buffer
                            _log.nest(2).trace("Try to prefetch spill '{0}'", prefetchDMA.opIdx_);
                            buffersPlusCurrentPrefetch.insert(prefetchDMA.buffer_);
                            buffersPlusPrefetch.insert(prefetchDMA.buffer_);
                        }

                        // skip if no new buffers added to be allocated
                        if (buffersNeedingAllocation.size() == buffersPlusCurrentPrefetch.size()) {
                            continue;
                        }

                        if (prefetchCount > barrierLimit) {
                            notAllocatedPrefetchOps.insert(prefetchDMA);
                            continue;
                        }

                        if (_executorPipelines[VPU::ExecutorKind::DMA_NN] > computeEndCycle) {
                            if (prefetchFront->activationSpill &&
                                prefetchFront->computeOpLevel + 1 == prefetchDMA.level_) {
                                // delay next DPU start only by prefetching its inputs
                            } else {
                                // do not delay next DPU
                                notAllocatedPrefetchOps.insert(prefetchDMA);
                                continue;
                            }
                        }

                        auto sortedPrefetchBuffers = sortUsedBuffers(buffersPlusPrefetch);
                        if (!_scan.canAlloc(sortedPrefetchBuffers)) {
                            notAllocatedPrefetchOps.insert(prefetchDMA);
                            continue;
                        }

                        if (prefetchDMA.buffer_ == nullptr) {
                            for (auto& buff : getNonAliveBuffersUsedByOperation(prefetchDMA.opIdx_)) {
                                _log.nest().trace("Mark prefetch buffer as alive, '{0}'", buff);
                                // special case for spilled reads
                                if (_opWritingToBuffer.find(buff) != _opWritingToBuffer.end()) {
                                    auto writerOp = retrieveBufferWriter(buff);
                                    auto executeOpIdx =
                                            _depsInfo.getIndex(writerOp->getBlock()
                                                                       ->getParent()
                                                                       ->getParentOfType<mlir::async::ExecuteOp>());
                                    scheduleSpilledOpBuffer(executeOpIdx, &buff);
                                }
                            }
                            schedulePrefetchOp(prefetchDMA.opIdx_);
                        } else {
                            _log.nest().trace("Mark prefetch spilled buffer as alive, '{0}'", prefetchDMA.buffer_);
                            _scan.handler().markAsAlive(prefetchDMA.buffer_);
                            scheduleSpilledOpBuffer(prefetchDMA.opIdx_, &prefetchDMA.buffer_);
                        }
                        buffersNeedingAllocation = buffersPlusCurrentPrefetch;
                        ++prefetchCount;
                    }
                }
            }
        } else {
            // case for non-DPU executor operations with dependencies
            _log.nest(1).trace("Update prefetch operations with currently scheduled ops");
            auto prefetchDataOps = prefetchFront->dataOps;
            prefetchSet newPrefetchDataOps;
            // remove allocated operations from prefetch set
            for (auto prefetchDMA : prefetchDataOps) {
                if (prefetchDMA.buffer_ == nullptr) {
                    for (auto& buff : getNonAliveBuffersUsedByOperation(prefetchDMA.opIdx_)) {
                        if (buffersNeedingAllocation.find(buff) != buffersNeedingAllocation.end()) {
                            _log.nest(2).trace("Operation '{0}' was allocated, removing from prefetch edges",
                                               prefetchDMA.opIdx_);
                            continue;
                        }
                        newPrefetchDataOps.insert(prefetchDMA);
                    }
                } else {
                    if (buffersNeedingAllocation.find(prefetchDMA.buffer_) != buffersNeedingAllocation.end()) {
                        _log.nest(2).trace("Operation '{0}' was allocated, removing from prefetch edges",
                                           prefetchDMA.opIdx_);
                        continue;
                    }
                    newPrefetchDataOps.insert(prefetchDMA);
                }
            }
            prefetchFront->dataOps = newPrefetchDataOps;
        }

        // if candidates were not prefetched during the current compute op due to
        // constraints of size or cycles move candidates to the next compute op
        auto nextFront = prefetchFront;
        ++nextFront;
        if (nextFront != _prefetchSchedule.end()) {
            for (auto& notAllocated : notAllocatedPrefetchOps) {
                nextFront->dataOps.insert(notAllocated);
            }
        }
    }
}

size_t FeasibleMemoryScheduler::getOperationEndCycle(operationIdxType opIdx, size_t nextScheduleCycle) {
    // Check if any of operation input dependencies have been scheduled in the same or previous
    // scheduler iteration. In such case earliestComputeBeginCycle might need to be adjusted
    // based on cycle completion of its input dependencies
    for (auto& heapElement : _cycleBeginHeap) {
        if (opIdx == heapElement.op_) {
            nextScheduleCycle = std::max(nextScheduleCycle, heapElement.cycleEnd_);
        }
    }
    for (auto& heapElement : _cycleEndHeap) {
        if (opIdx == heapElement.op_) {
            nextScheduleCycle = std::max(nextScheduleCycle, heapElement.cycleBegin_);
        }
    }
    for (auto& scheduledOp : _scheduledOps) {
        if (opIdx == scheduledOp.op_) {
            nextScheduleCycle = std::max(nextScheduleCycle, scheduledOp.cycleEnd_);
        }
    }

    return nextScheduleCycle;
}

size_t FeasibleMemoryScheduler::allocateBuffersAndInputOps(operationIdxType opIdx, Partitioner::Direction allocDir) {
    // retrieve op demand list - input ops
    auto usedBuffers = getNonAliveBuffersUsedByOperation(opIdx);
    auto demandList = getNonEmptyOpDemandList(opIdx, usedBuffers);
    size_t earliestComputeBeginCycle = getCurrentCycle(opIdx);
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
            auto nextAvailableCycle = scheduleSpilledOpBuffer(executeOpIdx, &val);
            earliestComputeBeginCycle = std::max(earliestComputeBeginCycle, nextAvailableCycle);
        }
    }

    // retrieve operation input's buffers
    for (auto inputIdx : demandList) {
        for (auto& val : getNonAliveBuffersUsedByOperation(inputIdx)) {
            buffersNeedingAllocation.insert(val);
            _log.nest().trace("Mark input op buffer as alive, '{0}'", val);
            _scan.handler().markAsAlive(val);
            // check if non-alive input buffer was spilled
            if (_opWritingToBuffer.find(val) != _opWritingToBuffer.end()) {
                // if so schedule the required spill read for it
                auto writerOp = retrieveBufferWriter(val);
                auto executeOpIdx = _depsInfo.getIndex(
                        writerOp->getBlock()->getParent()->getParentOfType<mlir::async::ExecuteOp>());
                auto nextAvailableCycle = scheduleSpilledOpBuffer(executeOpIdx, &val);
                earliestComputeBeginCycle = std::max(earliestComputeBeginCycle, nextAvailableCycle);
            }
        }
        auto nextAvailableCycle = scheduleInputOpForComputeOp(inputIdx);
        earliestComputeBeginCycle = std::max(earliestComputeBeginCycle, nextAvailableCycle);
    }

    // check if delayed by dependencies
    for (auto& dep : _depsInfo.getOpDeps(opIdx)) {
        earliestComputeBeginCycle = getOperationEndCycle(dep, earliestComputeBeginCycle);
    }

    // prefetch DMAs and add their buffers to be allocated
    allocatePrefetchOps(opIdx, earliestComputeBeginCycle, buffersNeedingAllocation);
    auto sortedBuffers = sortUsedBuffers(buffersNeedingAllocation);

    // allocate buffers using LinearScan
    _log.nest().trace("Allocate memory for the alive buffers");
    VPUX_THROW_UNLESS(_scan.alloc(sortedBuffers, false, allocDir), "Failed to statically allocate '{0}' memory",
                      _memKind);

    return earliestComputeBeginCycle;
}

size_t FeasibleMemoryScheduler::scheduleComputeOp(operationIdxType opIdx) {
    // Step 1: add to output result table
    _opOutputTable.insert(std::make_pair(opIdx, OpOutputInfo(EOpState::ACTIVE, _outDegreeTable[opIdx])));

    // Step 2: assign resources simultaneously
    auto earliestComputeBeginCycle = allocateBuffersAndInputOps(opIdx);

    // Step 3: update current cycles
    size_t nextAvailableCycle = earliestComputeBeginCycle + operationCycleCost(opIdx);
    updateCurrentCycleForExecutor(opIdx, nextAvailableCycle);

    // Step 4: schedule the compute op
    pushToCycleBeginHeap(HeapElement(opIdx, earliestComputeBeginCycle, nextAvailableCycle, EOpType::ORIGINAL_OP));

    if (!_prefetchSchedule.empty() && _prefetchSchedule.front().computeOpIdx == opIdx &&
        getExecutorType(opIdx) == VPU::ExecutorKind::NCE) {
        // preserve order of compute operations
        _prefetchSchedule.pop_front();
    }

    return nextAvailableCycle;
}

void FeasibleMemoryScheduler::scheduleAllPossibleReadyOpsAndUpdate() {
    // store scheduled op idx and end cycle
    SmallVector<std::pair<operationIdxType, size_t>> scheduledReadyOps;
    SmallVector<std::pair<operationIdxType, size_t>> scheduledNonComputeChainOps;
    bool DPUScheduled = false;
    _log.trace("Scheduling all possible ready ops");
    _log = _log.nest();

    // 1. schedule operation not belonging to main network compute chain as soon as they become
    // ready so that they execute in the next availible cycle since they are not prefetched
    for (auto& readyOpIdx : _nonComputeChainOps) {
        // Scheduling such operations can only happen once all input dependencies
        // (both data and compute ops) have already been executed. This is different
        // to standard compute op which as part of its scheduling can force scheduling
        // of needed data ops
        bool areDepsReady = true;
        for (auto& dep : _depsInfo.getOpDeps(readyOpIdx)) {
            if (_opOutputTable.find(dep) == _opOutputTable.end()) {
                areDepsReady = false;
                break;
            }
        }
        if (areDepsReady && isReadyComputeOperationSchedulable(readyOpIdx)) {
            _log.trace("Scheduling non compute chain op: '{0}'", readyOpIdx);
            auto computeOpEndCycle = scheduleComputeOp(readyOpIdx);
            scheduledNonComputeChainOps.push_back(std::make_pair(readyOpIdx, computeOpEndCycle));
        }
    }

    // TODO: heuristic for choosing next schedulable op

    // 2. schedule ready operations
    for (auto& readyOpIdx : _readyComputeOps) {
        if (isReadyComputeOperationSchedulable(readyOpIdx)) {
            // allocate only one DPU in an iteration
            if (getExecutorType(readyOpIdx) == VPU::ExecutorKind::NCE) {
                if (DPUScheduled) {
                    break;
                } else {
                    DPUScheduled = true;
                }
            }
            _log.trace("Scheduling ready op: '{0}'", readyOpIdx);
            auto computeOpEndCycle = scheduleComputeOp(readyOpIdx);
            scheduledReadyOps.push_back(std::make_pair(readyOpIdx, computeOpEndCycle));
        }
    }

    // 3. update ready lists by removing scheduled ops
    for (auto& scheduledOp : scheduledNonComputeChainOps) {
        _nonComputeChainOps.erase(scheduledOp.first);
        // delay all executors with non compute chain ops
        for (auto& pipeline : _executorPipelines) {
            pipeline.second = std::max(pipeline.second, scheduledOp.second);
            _log.nest().trace("Aligning executor pipeline {0} = {1}", pipeline.first, pipeline.second);
        }
    }
    for (auto& scheduledOp : scheduledReadyOps) {
        _readyComputeOps.erase(scheduledOp.first);
        // update pipelines
        if (hasBuffersInTargetMemoryKind(scheduledOp.first)) {
            // if operation in NNCMX align pipelines so that
            // operations are only allocated in the following cycles
            for (auto& pipeline : _executorPipelines) {
                pipeline.second = std::max(pipeline.second, scheduledOp.second);
                _log.nest().trace("Aligning executor pipeline {0} = {1}", pipeline.first, pipeline.second);
            }
        }
    }

    _log = _log.unnest();
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

    auto nextFront = _prefetchSchedule.begin();
    if (nextFront != _prefetchSchedule.end()) {
        nextFront->dataOps.insert(PrefetchDMA(evictionCandidate.bufferWriterIdx_,
                                              _bufferLevels[evictionCandidate.buffer_], evictionCandidate.buffer_));
    }

    _log.nest().trace("Mark buffer as dead, '{0}'", evictionCandidate.buffer_);
    _scan.handler().markAsDead(evictionCandidate.buffer_);

    _log.nest().trace("Free non alive buffers");
    _scan.freeNonAlive();
}

VPUIP::LayerOpInterface FeasibleMemoryScheduler::retrieveBufferWriter(mlir::Value buffer) {
    const auto valIt = _opWritingToBuffer.find(buffer);
    VPUX_THROW_UNLESS(valIt != _opWritingToBuffer.end(), "Buffer not scheduled yet, invalid spill candidate");
    return valIt->second;
}

size_t FeasibleMemoryScheduler::evictionPriority(mlir::Value buffer) {
    // TODO: E#21936 add other conditions such as:
    // pipelined, multiple outdegree (prefetch)

    // Eviction priority (highest evicted first):
    // (0) - buffers which are CMX Concatable
    // (1) - timestamp op buffers
    // (2) - buffers which are result of computeOp
    // (3) - buffers which are result of dataOp
    // (4) - buffers which are not going to be used by any ready compute ops

    auto writerOp = retrieveBufferWriter(buffer);
    auto executeOp = writerOp->getBlock()->getParent()->getParentOfType<mlir::async::ExecuteOp>();
    bool isBufferUsedByReadyOp = false;
    bool cmxConcatable = false;

    if (!isBufferUsedByReadyOp) {
        for (auto& readyOpIdx : _readyComputeOps) {
            auto op = _depsInfo.getExecuteOpAtIndex(readyOpIdx).getOperation();

            if (_liveRangeInfo.isBufferUsedByOp(buffer, op)) {
                isBufferUsedByReadyOp = true;
                break;
            }
        }
    }

    for (auto bufferUser : buffer.getUsers()) {
        if (mlir::isa<VPUIP::ConcatViewOp>(bufferUser)) {
            cmxConcatable = true;
            break;
        }
    }

    if (cmxConcatable) {
        return 0;
    } else if (mlir::isa<VPUIP::TimestampOp>(writerOp)) {
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

    auto first = evictionCandidates.begin();
    // first element has the smallest priority
    auto candidate = first;
    // try to find a DMA with a lower level, which will not be used by
    // the next DPU task, and may be prefetched during the next DPU
    size_t candidateLevel = std::numeric_limits<size_t>::min();
    if (_dataOpLevels.find(candidate->bufferWriterIdx_) != _dataOpLevels.end()) {
        candidateLevel = _dataOpLevels[candidate->bufferWriterIdx_];
    }
    while (first != evictionCandidates.end()) {
        if (_dataOpLevels.find(first->bufferWriterIdx_) != _dataOpLevels.end()) {
            if (_dataOpLevels[first->bufferWriterIdx_] > candidateLevel) {
                // select future data op
                candidate = first;
                candidateLevel = _dataOpLevels[first->bufferWriterIdx_];
            }
        }
        ++first;
    }

    return *candidate;
}

void FeasibleMemoryScheduler::forceScheduleActiveOpEviction() {
    _log.trace("Unable to schedule an operation, forcing dynamic spill");
    // retrieve the alive buffers
    auto aliveBuffers = _scan.handler().getAliveValues();
    if (aliveBuffers.empty()) {
        _log.error("Scheduler cannot schedule anything and there is no buffer to spill");
        _log.error("Ready operations:");
        for (auto& readyOp : _readyComputeOps) {
            // Get operation demand list - operations which were not scheduled yet,
            // but need to be scheduled to produce required buffers
            auto usedBuffers = getNonAliveBuffersUsedByOperation(readyOp);
            auto opAndDemandList = getNonEmptyOpDemandList(readyOp, usedBuffers);
            opAndDemandList.insert(readyOp);

            mlir::DenseSet<mlir::Value> buffersNeedingAllocation;
            // Retrieve all buffers that need to be allocated to schedule this operation
            for (auto opIdx : opAndDemandList) {
                for (auto val : getNonAliveBuffersUsedByOperation(opIdx)) {
                    buffersNeedingAllocation.insert(val);
                }
            }
            // Calculate total size needed to allocate all those buffers
            size_t totalSize = 0;
            for (auto& buf : buffersNeedingAllocation) {
                totalSize += _scan.handler().getSize(buf);
            }

            _log.nest().error("opIdx: {0}, size demand: {1}, op: {2}", readyOp, totalSize,
                              _depsInfo.getExecuteOpAtIndex(readyOp));
        }

        VPUX_THROW("Scheduler failure, cannot schedule anything and there is no buffer to spill");
    }

    // select a candidate op to be spilled
    auto evictionCandidate = chooseCandidateForEviction(aliveBuffers);
    _log.nest().trace("Candidate selected for eviction {0} {1}", evictionCandidate.bufferWriterIdx_,
                      evictionCandidate.buffer_);

    // free the memory space by freeing the op output buffer
    evictActiveOp(evictionCandidate);
    _log.nest().trace("Candidate evicted and spilled");

    auto scheduleCycle = getCurrentCycle(evictionCandidate.bufferWriterIdx_);
    // check if delayed by spilling operation
    scheduleCycle = getOperationEndCycle(evictionCandidate.bufferWriterIdx_, scheduleCycle);
    // find operation end cycle
    auto nextAvailableCycle = scheduleCycle + operationCycleCost(evictionCandidate.bufferWriterIdx_, true);
    // add with a spilled write state
    pushToCycleBeginHeap(HeapElement(evictionCandidate.bufferWriterIdx_, scheduleCycle, nextAvailableCycle,
                                     EOpType::IMPLICIT_SPILL_WRITE_OP, evictionCandidate.buffer_));
    // update current cycle directly
    updateCurrentCycleForExecutor(evictionCandidate.bufferWriterIdx_, nextAvailableCycle, true);
}

void FeasibleMemoryScheduler::populateScheduledOps(HeapElement& scheduledOp) {
    SmallVector<IntervalInfo> outputIntervals;
    SmallVector<IntervalInfo> inputIntervals;
    // store scheduled information
    if (scheduledOp.isSpillWriteOp()) {
        // special case for a spill write with deallocation
        IntervalInfo interval;
        // retrieve and store operation addresses
        interval.begin_ = checked_cast<size_t>(_scan.handler().getAddress(scheduledOp.spillBuffer_));
        interval.end_ = interval.begin_ + checked_cast<size_t>(_scan.handler().getSize(scheduledOp.spillBuffer_));
        interval.buffer_ = scheduledOp.spillBuffer_;
        // SPILL WRITE has only input resource
        inputIntervals.push_back(interval);
        // deallocate only after addresses stored
        _log.nest().trace("Deallocate, '{0}'", scheduledOp.spillBuffer_);
        _scan.handler().deallocate(scheduledOp.spillBuffer_);
    } else {
        // retrieve interval information, operation can have multiple output buffers
        auto* bodyBlock = &_depsInfo.getExecuteOpAtIndex(scheduledOp.op_).body().front();
        for (auto& op : bodyBlock->getOperations()) {
            if (mlir::isa<VPUIP::LayerOpInterface>(op)) {
                auto layerOp = mlir::dyn_cast<VPUIP::LayerOpInterface>(op);

                if (scheduledOp.isOriginalOp()) {
                    // Track input resources. SPILL READ has only output resource
                    mlir::DenseSet<mlir::Value> inputBuffers;
                    for (auto input : layerOp.getInputs()) {
                        const auto type = input.getType().dyn_cast<vpux::NDTypeInterface>();
                        if (type == nullptr || type.getMemoryKind() != _memKind) {
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
                    if (type == nullptr || type.getMemoryKind() != _memKind) {
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
                    if (_opWritingToBuffer.find(rootBuffer) == _opWritingToBuffer.end()) {
                        // do not overwrite to preserve correct dependencies
                        _opWritingToBuffer[rootBuffer] = layerOp;
                    }
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
    scheduled.outputResourceInfo_ = outputIntervals;
    scheduled.inputResourceInfo_ = inputIntervals;
    scheduled.cycleBegin_ = scheduledOp.cycleBegin_;
    scheduled.cycleEnd_ = scheduledOp.cycleEnd_;
    scheduled.isDataOp_ = isDataOp(scheduledOp.op_);
    scheduled.freeCmx_ = _scan.totalFreeSize();
    scheduled.executor = scheduledOp.isSpillOp() ? VPU::ExecutorKind::DMA_NN : getExecutorType(scheduledOp.op_);
    scheduled.isNonComputeChain = isNonComputeChainOp(scheduledOp.op_);
    _scheduledOps.push_back(scheduled);
    _log.trace("Scheduled op: '{0}'", scheduled.op_);
}

void FeasibleMemoryScheduler::clearLists() {
    _readyComputeOps.clear();  // ready compute operations
    _readyDataOps.clear();     // ready data inputs (->CMX)
}

bool FeasibleMemoryScheduler::init() {
    _log.trace("Feasible Memory Scheduler init()");
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
    initializeReadyLists();
    schedulingLoop();

    return true;
}

void FeasibleMemoryScheduler::schedulingLoop() {
    // scheduling loop, loop until all output ops are scheduled
    while (!_outputOps.empty()) {
        if (!_cycleBeginHeap.empty()) {
            // if operation exist in cycle begin heap: move to cycle end heap
            _log.trace("Popping from cycle begin heap");

            // 1. schedule first operation from cycle begin heap
            HeapElement firstOp = popFromCycleBeginHeap();
            // 2. schedule operation by adding to _scheduledOps
            populateScheduledOps(firstOp);
            // 3. add operation to cycle end heap
            pushToCycleEndHeap(HeapElement(firstOp.op_, firstOp.cycleEnd_, firstOp.cycleBegin_, firstOp.opType_));
            // 4. decrease outputs if output operation scheduled
            if (_outputOps.find(firstOp.op_) != _outputOps.end()) {
                _outputOps.erase(firstOp.op_);
            }
        } else {
            // try to schedule new operations
            _log.trace("Try to schedule new operations");

            // 1. unschedule all operations from cycle end heap
            //  - free memory of consumed buffers
            //  - unlock new ready operations
            unscheduleAllCompletingOps();

            // 2. schedule ready operations
            //  - allocate compute operations along with data dependencies
            //  - update executor pipelines
            scheduleAllPossibleReadyOpsAndUpdate();

            // 3. if no operation was added to cycle begin heap after scheduling
            //  - unable to schedule an operation, perform dynamic spill
            if (_cycleBeginHeap.empty()) {
                forceScheduleActiveOpEviction();
            }
        }
    }
}

SmallVector<FeasibleMemoryScheduler::ScheduledOpInfo> FeasibleMemoryScheduler::generateSchedule(
        scheduleWithPrefetch prefetchSchedule) {
    // iteration with prefetching edges
    if (!prefetchSchedule.empty()) {
        _scan.handler().markAllBuffersAsDead();
        _prefetchSchedule = prefetchSchedule;
    }

    // start the memory scheduler
    init();

    // schedule quality based on cycles (cycles start from 1)
    size_t DPUCycles = 1;
    size_t DMACycles = 1;
    size_t totalCycles = 0;

    _log.trace("Generated Schedule");
    _log = _log.nest();
    for (const auto& op : _scheduledOps) {
        auto execOp = _depsInfo.getExecuteOpAtIndex(op.op_);
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

        if (!prefetchSchedule.empty()) {
            if (execOp->hasAttr("exceedingNNCMX")) {
                // remove pass specific attributes after complete allocation
                execOp->removeAttr("exceedingNNCMX");
            }
        }

        auto cycleInfo = "cycles = " + std::to_string(op.cycleBegin_) + " -> " + std::to_string(op.cycleEnd_);
        _log.trace("op = '{0}'\t executor = '{1}'\t type = '{2}'\t '{3}'\t inputs = '{4}' outputs = '{5}' \t free = "
                   "'{6}'\t name = '{7}'",
                   op.op_, op.executor, op.opTypeName(), cycleInfo, inputResourceInfo, outputResourceInfo, op.freeCmx_,
                   execOp.getLoc());

        if (op.executor == VPU::ExecutorKind::DMA_NN) {
            auto cycleDiff = op.cycleBegin_ - DMACycles;
            if (cycleDiff > 0) {
                _log.nest().trace("--- DMA STALL ({0} cycles) ---", cycleDiff);
            }
            DMACycles = op.cycleEnd_;
            // check if op is waiting for it's dependency to finish
            // if so stall can not be eliminated
        } else if (op.executor == VPU::ExecutorKind::NCE) {
            auto cycleDiff = op.cycleBegin_ - DPUCycles;
            if (cycleDiff != 0) {
                _log.nest().trace("--- DPU STALL ({0} cycles) ---", cycleDiff);
            }
            DPUCycles = op.cycleEnd_;
            // check if op is waiting for it's dependency to finish
            // if so stall can not be eliminated
        }
        totalCycles = std::max(totalCycles, op.cycleEnd_);
    }
    _log = _log.unnest();
    _log.trace("Total Cycles = {0}", totalCycles);

    return _scheduledOps;
}
