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

#include "vpux/compiler/dialect/VPURT/barrier_resource_state.hpp"

using namespace vpux::VPURT;

//
// BarrierResourceState
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

BarrierResourceState::BarrierResourceState(): _barrierReference(), _globalAvailableProducerSlots() {
}

BarrierResourceState::BarrierResourceState(size_t barrierCount, size_t maximumProducerSlotCount)
        : _barrierReference(), _globalAvailableProducerSlots() {
    init(barrierCount, maximumProducerSlotCount);
}

// Initialises the barrier resource state with the number of available barrier and the maximum allowable producers per
// barrier
void BarrierResourceState::init(const size_t barrierCount, const size_t maximumProducerSlotCount) {
    VPUX_THROW_UNLESS((barrierCount && maximumProducerSlotCount),
                      "Error number of barrier and/or producer slot count is 0");
    _globalAvailableProducerSlots.clear();
    _barrierReference.clear();

    availableSlotsIteratorType hint = _globalAvailableProducerSlots.end();
    for (size_t barrierId = 0UL; barrierId < barrierCount; barrierId++) {
        hint = _globalAvailableProducerSlots.insert(hint,
                                                    availableSlotKey(maximumProducerSlotCount, size_t(barrierId)));
        _barrierReference.push_back(hint);
    }
}

BarrierResourceState::constAvailableslotsIteratorType BarrierResourceState::findBarrierWithAvailableSlots(
        size_t slotDemand) {
    availableSlotKey key(slotDemand);

    // Return an iterator to the first barrier ID that is found which has the available requested slots
    constAvailableslotsIteratorType itr = _globalAvailableProducerSlots.lower_bound(key);
    constAvailableslotsIteratorType retItr = itr;

    // The start condition is the first barrier ID that is found which has the available requested slots
    // Then iterate from this point to find a barrier that is currently unused
    for (; (itr != _globalAvailableProducerSlots.end()) && (itr->isBarrierInUse()); ++itr) {
    }
    if (itr != _globalAvailableProducerSlots.end()) {
        retItr = itr;
    };

    return retItr;
}
// Returns true if there is a barrier with available producer slots
bool BarrierResourceState::hasBarrierWithAvailableSlots(size_t slotDemand) {
    availableSlotKey key(slotDemand);

    auto retItr = findBarrierWithAvailableSlots(slotDemand);

    return retItr != _globalAvailableProducerSlots.end();
}

// Precondition: hasBarrierWithAvailableSlots(slotDemand) is true
// Assigns the requested producers slots (resource) from a barrier ID when a task is scheduled by the main list
// scheduler
size_t BarrierResourceState::assignBarrierSlots(size_t slotDemand) {
    availableSlotKey key(slotDemand);

    auto itr = findBarrierWithAvailableSlots(slotDemand);

    if ((itr == _globalAvailableProducerSlots.end()) || (itr->_availableProducerSlots < slotDemand)) {
        return invalidBarrier();
    }

    size_t barrierId = itr->_barrier;
    return assignBarrierSlots(barrierId, slotDemand) ? barrierId : invalidBarrier();
}

// Assigns the requested producers slots (resource) from a barrier ID when a task is scheduled by the main list
// scheduler
bool BarrierResourceState::assignBarrierSlots(size_t barrierId, size_t slotDemand) {
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
bool BarrierResourceState::unassignBarrierSlots(size_t barrierId, size_t slotDemand) {
    VPUX_THROW_UNLESS((barrierId <= _barrierReference.size()),
                      "Invalid barrierId {0} supplied, it must be from from 0 to {1}", _barrierReference.size() - 1);
    availableSlotsIteratorType itr = _barrierReference[barrierId];
    size_t freeSlots = (itr->_availableProducerSlots) + slotDemand;

    itr = update(itr, freeSlots);
    return (itr != _globalAvailableProducerSlots.end());
}

size_t BarrierResourceState::invalidBarrier() {
    return std::numeric_limits<size_t>::max();
}

// NOTE: will also update _barrierReference
void BarrierResourceState::update(size_t barrierId, size_t updatedAvailableProducerSlots) {
    VPUX_THROW_UNLESS((barrierId <= _barrierReference.size()),
                      "Invalid barrierId {0} supplied, it must be from 0 to {1}", _barrierReference.size() - 1);
    availableSlotsIteratorType itr = _barrierReference[barrierId];
    update(itr, updatedAvailableProducerSlots);
}

// Updates the state of a barrier with current available producers slots
// This should be called whenever a task is scheduled/unscheduled by the main list scheduler
BarrierResourceState::availableSlotsIteratorType BarrierResourceState::update(availableSlotsIteratorType itr,
                                                                              size_t updatedAvailableProducerSlots) {
    VPUX_THROW_UNLESS(itr != _globalAvailableProducerSlots.end(), "Invalid _globalAvailableProducerSlots iterator");

    availableSlotKey key = *itr;
    key._availableProducerSlots = updatedAvailableProducerSlots;
    _globalAvailableProducerSlots.erase(itr);

    itr = (_globalAvailableProducerSlots.insert(key)).first;
    VPUX_THROW_UNLESS(itr != _globalAvailableProducerSlots.end(), "Invalid _globalAvailableProducerSlots iterator");

    _barrierReference[(itr->_barrier)] = itr;
    return itr;
}
