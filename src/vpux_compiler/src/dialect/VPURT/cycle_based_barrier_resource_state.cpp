//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPURT/cycle_based_barrier_resource_state.hpp"

using namespace vpux::VPURT;

//
// CycleBasedBarrierResourceState
//
// This class provided a mechanism to maintain/assign/unassign barrier resources
// to the main barrier list scheduler.
//
// The barrier scheduler can only schedule tasks if there are available barrier resources
// and these resources need to be assign/unassign when tasks are scheduled/unscheduled.
//
// VPU hardware only has a finite number of barriers, 8 per cluster. The Barrier scheduler class
// ensures that the number of active barriers does not exceed the available
// physical barriers for the target platform.
//
// The hardware allows a finite producer count of 256 for each of the update barriers.
// This means that multiple tasks can update the same active barrier. This is incorporated into the
// barrier resource model by using the term "barrier slots".
// In addition to the upper bound of available barriers it is assumed that each of these barriers has
// a maximum of 256 slots. The barrier demand is expressed as the number of slots required.
// In the context of VPU hardware the number of slots for a DPU tasks are the DPU workloads
// and for a DMA/UPA tasks it is 1.

//
// Constructor
//

CycleBasedBarrierResourceState::CycleBasedBarrierResourceState(): _globalAvailableProducerSlots(), _barrierReference() {
}

// Initialises the barrier resource state with the number of available barrier and the maximum allowable producers per
// barrier
void CycleBasedBarrierResourceState::init(
        const size_t barrierCount, const size_t maximumProducerSlotCount, SmallVector<VPURT::TaskOp> orderedTasks,
        std::map<TaskQueueType, SmallVector<VPURT::TaskOp>> orderedTasksByCycleStart,
        std::map<mlir::Operation*, std::set<mlir::Operation*>> taskConsumerMapOriginal) {
    VPUX_THROW_UNLESS((barrierCount && maximumProducerSlotCount),
                      "Barrier scheduler error: The number of physical barrier and/or producer slot per barrier is 0");
    _globalAvailableProducerSlots.clear();
    _barrierReference.clear();
    _barrierProducers.clear();
    _barrierConsumers.clear();
    _physicalToVirtual.clear();

    availableSlotsIteratorType hint = _globalAvailableProducerSlots.end();
    for (size_t barrierId = 0UL; barrierId < barrierCount; barrierId++) {
        hint = _globalAvailableProducerSlots.insert(hint,
                                                    availableSlotKey(maximumProducerSlotCount, size_t(barrierId)));
        _barrierReference.push_back(hint);
    }
    _barrierProducers.resize(barrierCount);
    _barrierConsumers.resize(barrierCount);
    _physicalToVirtual.resize(barrierCount, -1);
    _orderedTasks = orderedTasks;
    _orderedTasksByCycleStart = orderedTasksByCycleStart;
    _taskConsumerMapOriginal = taskConsumerMapOriginal;
}

CycleBasedBarrierResourceState::constAvailableslotsIteratorType
CycleBasedBarrierResourceState::findUnusedBarrierWithAvailableSlots(size_t slotDemand) {
    availableSlotKey key(slotDemand);
    // Return an iterator to the first barrier ID that is found which has the available requested slots
    constAvailableslotsIteratorType itr = _globalAvailableProducerSlots.lower_bound(key);

    // The start condition is the first barrier ID that is found which has the available requested slots
    // Then iterate from this point to find a barrier that is currently unused
    // The definition of an used barrier is both producer and consumer is empty at the moment
    for (; (itr != _globalAvailableProducerSlots.end()); ++itr) {
        if (_barrierConsumers[itr->_barrier].empty() && (!itr->isBarrierInUse())) {
            return itr;
        }
    }
    return _globalAvailableProducerSlots.end();
}

// If unused barrier is not found, find a barrier by following principle
// 1) virtual ID is larger than task's wait barriers. It makes sure the virtual ID of update barrier is larger than
// wait barrier for a task. For example, if task A waits on virtual barrier 2, we don't want it updates virtual
// barrier 1 because it doesn't make sense and usually will make simulation fails
// 2) The consumer list of selected barrier doesn't include a scheduled task which has same executor kind as current
// task
// 3) The cycle overlap between producer and consumer is minimum for the selected barrier
CycleBasedBarrierResourceState::constAvailableslotsIteratorType
CycleBasedBarrierResourceState::findBarrierWithMinimumCycleDelay(
        size_t slotDemand, size_t latestWaitBarrier, mlir::Operation* producerTask,
        SmallVector<llvm::BitVector>& configureBarrierOpWaitMap,
        SmallVector<llvm::BitVector>& configureBarrierOpUpdateMap) {
    availableSlotKey key(slotDemand);

    // Return an iterator to the first barrier ID that is found which has the available requested slots
    constAvailableslotsIteratorType itr = _globalAvailableProducerSlots.lower_bound(key);
    constAvailableslotsIteratorType retItr = _globalAvailableProducerSlots.end();

    int64_t overlapCycle = std::numeric_limits<int64_t>::max();
    auto executorKind = mlir::dyn_cast<VPURT::TaskOp>(producerTask).getExecutorKind();
    for (; (itr != _globalAvailableProducerSlots.end()); ++itr) {
        if (_physicalToVirtual[itr->_barrier] > static_cast<int64_t>(latestWaitBarrier)) {
            bool invalidBarrier = false;
            auto virtualID = _physicalToVirtual[itr->_barrier];
            auto origProducerTasks = configureBarrierOpWaitMap[virtualID].set_bits();
            auto origConsumerTasks = configureBarrierOpUpdateMap[virtualID].set_bits();
            size_t currentFirstStart = std::numeric_limits<size_t>::max();
            size_t currentLastEnd = getTaskEndCycle(producerTask);

            for (auto producer : origProducerTasks) {
                auto producerEndCycle = getTaskEndCycle(_orderedTasks[producer].getOperation());
                if (currentLastEnd < producerEndCycle) {
                    currentLastEnd = producerEndCycle;
                }
            }

            for (auto consumer : origConsumerTasks) {
                if ((_orderedTasks[consumer].getExecutorKind() == executorKind) &&
                    (_orderedTasks[consumer]->hasAttr(schedulingNumberAttrName))) {
                    invalidBarrier = true;
                    break;
                }

                auto consumerStartCycle = getTaskStartCycle(_orderedTasks[consumer].getOperation());
                if (consumerStartCycle < currentFirstStart) {
                    currentFirstStart = consumerStartCycle;
                }
            }

            for (auto consumer : _taskConsumerMapOriginal[producerTask]) {
                auto consumerStartCycle = getTaskStartCycle(consumer);
                if (consumerStartCycle < currentFirstStart) {
                    currentFirstStart = consumerStartCycle;
                }
            }

            if (!invalidBarrier) {
                if ((static_cast<int64_t>(currentLastEnd) - static_cast<int64_t>(currentFirstStart)) < overlapCycle) {
                    overlapCycle = static_cast<int64_t>(currentLastEnd) - static_cast<int64_t>(currentFirstStart);
                    retItr = itr;
                }
            }
        }
    }

    return retItr;
}

// If scheduled tasks are all finished and we can't find a barrier that satisfies the requirement, it means that
// scheduling is blocked due to current dependency in the IR.
// Then we need to adjust the consumer task of an active barrier to make it become unused.
// Adjustment is replacing the active consumer with another task which has same executor kind and has already been
// scheduled.

// Find a candidate barrier by the following principle
// 1) Producers are finished
// 2) Only a consumer is still active
// 3) The cycle overlap between producer and consumer is minimum after adjusting consumer task for the selected barrier
bool CycleBasedBarrierResourceState::createUnusedBarrierByAdjustingConsumer(
        SmallVector<llvm::BitVector>& configureBarrierOpWaitMap,
        SmallVector<llvm::BitVector>& configureBarrierOpUpdateMap,
        SmallVector<llvm::BitVector>& configureTaskOpWaitMap) {
    auto retItr = _globalAvailableProducerSlots.end();

    int64_t overlapCycle = std::numeric_limits<int64_t>::max();
    constAvailableslotsIteratorType itrBestChoice = _globalAvailableProducerSlots.end();
    size_t blockConsumerIdBestChoice = 0;
    size_t newConsumerIdBestChoice = 0;
    constAvailableslotsIteratorType itr = _globalAvailableProducerSlots.begin();
    for (; (itr != _globalAvailableProducerSlots.end()); ++itr) {
        if ((_barrierConsumers[itr->_barrier].size() == 1) && (!itr->isBarrierInUse())) {
            auto blockConsumer = *(_barrierConsumers[itr->_barrier].begin());
            auto blockConsumerId = getTaskUniqueID(blockConsumer);
            auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(blockConsumer);
            auto barrierVirtualID = _physicalToVirtual[itr->_barrier];

            std::set<vpux::VPURT::TaskOp> newConsumerCandidates;
            auto dmaTaskQueueTypes = getDMATaskQueueType(taskOp);
            // When we have dual DMA engines, a DMA taskOp can use both engines before unrolling.
            // For such case, replacing it with another DMA task which executes before is not safe for its wait barrier.
            // For example, barrier A is consumed by DMA-1 which uses dual DMA engines. There's a DMA-0 executes before
            // DMA-1 but only use engine 0. Then if we replace DMA-1 with DMA-0 for barrier A, we can't garentee a part
            // of DMA-1 (uses engine 1) starts after barrier A is ready for consuming.
            if (dmaTaskQueueTypes.hasValue() && (dmaTaskQueueTypes.getValue().size() == 1)) {
                for (auto& queueType : dmaTaskQueueTypes.getValue()) {
                    vpux::VPURT::TaskOp newConsumer =
                            findPreviousScheduledTask(taskOp, _orderedTasksByCycleStart[queueType]);
                    if (newConsumer) {
                        newConsumerCandidates.insert(newConsumer);
                    }
                }
            } else {
                auto taskQueueType = getTaskQueueType(taskOp);
                vpux::VPURT::TaskOp newConsumer =
                        findPreviousScheduledTask(taskOp, _orderedTasksByCycleStart[taskQueueType]);
                if (newConsumer) {
                    newConsumerCandidates.insert(newConsumer);
                }
            }

            int64_t lastEnd = 0;
            auto producers = configureBarrierOpWaitMap[barrierVirtualID];
            for (auto prod = producers.set_bits_begin(); prod != producers.set_bits_end(); prod++) {
                auto producerEndCycle = getTaskEndCycle(_orderedTasks[*prod].getOperation());
                if (static_cast<int64_t>(producerEndCycle) > lastEnd) {
                    lastEnd = producerEndCycle;
                }
            }

            for (auto newConsumer : newConsumerCandidates) {
                auto newConsumerId = getTaskUniqueID(newConsumer.getOperation());
                auto newConsumerStartCycle = getTaskStartCycle(newConsumer.getOperation());
                if ((lastEnd - static_cast<int64_t>(newConsumerStartCycle)) < overlapCycle) {
                    overlapCycle = (lastEnd - newConsumerStartCycle);
                    itrBestChoice = itr;
                    blockConsumerIdBestChoice = blockConsumerId;
                    newConsumerIdBestChoice = newConsumerId;
                }
            }
        }
    }

    if (itrBestChoice != _globalAvailableProducerSlots.end()) {
        auto selectedBarrierVirtualID = _physicalToVirtual[itrBestChoice->_barrier];
        _barrierConsumers[itrBestChoice->_barrier].clear();
        configureBarrierOpUpdateMap[selectedBarrierVirtualID].reset(static_cast<unsigned>(blockConsumerIdBestChoice));
        configureBarrierOpUpdateMap[selectedBarrierVirtualID].set(static_cast<unsigned>(newConsumerIdBestChoice));
        configureTaskOpWaitMap[blockConsumerIdBestChoice].reset(static_cast<unsigned>(selectedBarrierVirtualID));
        configureTaskOpWaitMap[newConsumerIdBestChoice].set(static_cast<unsigned>(selectedBarrierVirtualID));
        retItr = itrBestChoice;
    }

    return retItr != _globalAvailableProducerSlots.end();
}

// Precondition: findBarrierMeetRequirement() is true
// Assigns the requested producers slots (resource) from a barrier ID when a task is scheduled by the main list
// scheduler
size_t CycleBasedBarrierResourceState::assignBarrierSlots(size_t slotDemand, mlir::Operation* producerTask,
                                                          size_t latestWaitBarrier, size_t& virtualId,
                                                          SmallVector<llvm::BitVector>& configureBarrierOpWaitMap,
                                                          SmallVector<llvm::BitVector>& configureBarrierOpUpdateMap) {
    auto itr = findUnusedBarrierWithAvailableSlots(slotDemand);
    if (itr == _globalAvailableProducerSlots.end()) {
        itr = findBarrierWithMinimumCycleDelay(slotDemand, latestWaitBarrier, producerTask, configureBarrierOpWaitMap,
                                               configureBarrierOpUpdateMap);
    }

    if ((itr == _globalAvailableProducerSlots.end()) || (itr->_availableProducerSlots < slotDemand)) {
        return invalidBarrier();
    }

    size_t barrierId = itr->_barrier;

    if (itr->isBarrierInUse() || (!_barrierConsumers[barrierId].empty())) {
        virtualId = _physicalToVirtual[barrierId];
    } else {
        _physicalToVirtual[barrierId] = virtualId;
    }

    if (assignBarrierSlots(barrierId, slotDemand)) {
        _barrierProducers[barrierId].insert(producerTask);

        for (auto task : _taskConsumerMapOriginal[producerTask]) {
            _barrierConsumers[barrierId].insert(task);
        }

        return barrierId;
    } else {
        return invalidBarrier();
    }
}

// Assigns the requested producers slots (resource) from a barrier ID when a task is scheduled by the main list
// scheduler
bool CycleBasedBarrierResourceState::assignBarrierSlots(size_t barrierId, size_t slotDemand) {
    VPUX_THROW_UNLESS((barrierId <= _barrierReference.size()),
                      "Invalid barrierId {0} supplied, it must be from 0 to {1}", barrierId,
                      _barrierReference.size() - 1);
    availableSlotsIteratorType itr = _barrierReference[barrierId];

    VPUX_THROW_UNLESS((itr->_availableProducerSlots) >= slotDemand,
                      "Error the available producer slots for barrier ID {0} is {1}, which is less than the requested "
                      "slots demand {2}",
                      barrierId, itr->_availableProducerSlots, slotDemand);

    size_t remainingSlots = (itr->_availableProducerSlots) - slotDemand;

    itr = update(itr, remainingSlots);
    return (itr != _globalAvailableProducerSlots.end());
}

// Releases the producers slots (resource) from a barrier ID when a task is unscheduled by the main list scheduler
bool CycleBasedBarrierResourceState::unassignBarrierSlots(size_t barrierId, size_t slotDemand, mlir::Operation* op) {
    VPUX_THROW_UNLESS((barrierId <= _barrierReference.size()),
                      "Invalid barrierId {0} supplied, it must be from from 0 to {1}", _barrierReference.size() - 1);
    availableSlotsIteratorType itr = _barrierReference[barrierId];
    size_t freeSlots = (itr->_availableProducerSlots) + slotDemand;

    itr = update(itr, freeSlots);
    _barrierProducers[barrierId].erase(op);
    return (itr != _globalAvailableProducerSlots.end());
}

size_t CycleBasedBarrierResourceState::invalidBarrier() {
    return std::numeric_limits<size_t>::max();
}

// NOTE: will also update _barrierReference
void CycleBasedBarrierResourceState::update(size_t barrierId, size_t updatedAvailableProducerSlots) {
    VPUX_THROW_UNLESS((barrierId <= _barrierReference.size()),
                      "Invalid barrierId {0} supplied, it must be from 0 to {1}", _barrierReference.size() - 1);
    availableSlotsIteratorType itr = _barrierReference[barrierId];
    update(itr, updatedAvailableProducerSlots);
}

// Updates the state of a barrier with current available producers slots
// This should be called whenever a task is scheduled/unscheduled by the main list scheduler
CycleBasedBarrierResourceState::availableSlotsIteratorType CycleBasedBarrierResourceState::update(
        availableSlotsIteratorType itr, size_t updatedAvailableProducerSlots) {
    VPUX_THROW_UNLESS(itr != _globalAvailableProducerSlots.end(), "Invalid _globalAvailableProducerSlots iterator");

    availableSlotKey key = *itr;
    key._availableProducerSlots = updatedAvailableProducerSlots;
    _globalAvailableProducerSlots.erase(itr);

    itr = (_globalAvailableProducerSlots.insert(key)).first;
    VPUX_THROW_UNLESS(itr != _globalAvailableProducerSlots.end(), "Invalid _globalAvailableProducerSlots iterator");

    _barrierReference[(itr->_barrier)] = itr;
    return itr;
}

void CycleBasedBarrierResourceState::updateBarrierConsumer(mlir::Operation* task, size_t barrierId) {
    size_t num = _barrierConsumers[barrierId].erase(task);
    VPUX_THROW_UNLESS(num, "task {0} is not found in barrier consumer list", task);
}

size_t CycleBasedBarrierResourceState::getTaskUniqueID(mlir::Operation* task) {
    return checked_cast<size_t>(
            mlir::dyn_cast<VPURT::TaskOp>(task)->getAttr(uniqueIdAttrName).cast<mlir::IntegerAttr>().getInt());
}

vpux::VPURT::TaskOp CycleBasedBarrierResourceState::findPreviousScheduledTask(
        vpux::VPURT::TaskOp task, SmallVector<vpux::VPURT::TaskOp>& orderedTasksByCycleStart) {
    auto itr = std::find(orderedTasksByCycleStart.rbegin(), orderedTasksByCycleStart.rend(), task);
    while ((itr != orderedTasksByCycleStart.rend()) && (!(*itr)->hasAttr(schedulingNumberAttrName))) {
        itr++;
    }

    if (itr != orderedTasksByCycleStart.rend()) {
        return *itr;
    }

    return nullptr;
}

size_t CycleBasedBarrierResourceState::getTaskStartCycle(mlir::Operation* task) {
    return checked_cast<size_t>(
            mlir::dyn_cast<VPURT::TaskOp>(task)->getAttr(cycleBegin).cast<mlir::IntegerAttr>().getInt());
}

size_t CycleBasedBarrierResourceState::getTaskEndCycle(mlir::Operation* task) {
    return checked_cast<size_t>(
            mlir::dyn_cast<VPURT::TaskOp>(task)->getAttr(cycleEnd).cast<mlir::IntegerAttr>().getInt());
}
