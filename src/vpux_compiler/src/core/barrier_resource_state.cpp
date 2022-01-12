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

#include "vpux/compiler/core/barrier_resource_state.hpp"

using namespace vpux::VPURT;

//
// BarrierResourceState
//
// This class provided a mechanism to maintain/assign/unassign barrier resources
// to the main barrier list scheduler.
//
// The barrier scheduler can only schedule tasks if there are available barrier resources
// and these rescources need to be assign/unassign when tasks are scheduled/unscheduled.
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
// In the context of VPU hardware the number of slots for a DPU tasks are the DPU worklaods
// and for a DMA/UPA tasks it is 1.

//
// Constructor
//

BarrierResourceState::BarrierResourceState(): _barrierReference(), _availableProducerSlots() {
}

BarrierResourceState::BarrierResourceState(size_t barrierCount, size_t maximumProducerSlotCount)
        : _barrierReference(), _availableProducerSlots() {
    init(barrierCount, maximumProducerSlotCount);
}

// Initialises the barrier resource state with the number of available barrier and the maximum allowable producers per
// barrier
void BarrierResourceState::init(const size_t barrierCount, const producerSlotsType maximumProducerSlotCount) {
    VPUX_THROW_UNLESS((barrierCount && maximumProducerSlotCount),
                      "Error number of barrier and/or producer slot count is 0");
    _availableProducerSlots.clear();
    _barrierReference.clear();
    _barrierUsers.clear();

    availableSlotsIteratorType hint = _availableProducerSlots.end();
    for (size_t barrierId = 1UL; barrierId <= barrierCount; barrierId++) {
        hint = _availableProducerSlots.insert(hint, availableSlotKey(maximumProducerSlotCount, barrierType(barrierId)));
        _barrierReference.push_back(hint);
    }
    _barrierUsers.resize(barrierCount);
}

// Returns true if there is a barrier with available producer slots
bool BarrierResourceState::hasBarrierWithSlots(producerSlotsType slotDemand) const {
    availableSlotKey key(slotDemand);
    constAvailableslotsIteratorType itr = _availableProducerSlots.lower_bound(key);
    constAvailableslotsIteratorType retItr = itr;

    // prefer a unused barrier to satisfy this slot demand //
    for (; (itr != _availableProducerSlots.end()) && (itr->isBarrierInUse()); ++itr) {
    }
    if (itr != _availableProducerSlots.end()) {
        retItr = itr;
    };

    return retItr != _availableProducerSlots.end();
}

// Precondition: hasBarrierWithSlots(slotDemand) is true
// Assigns the requested producers slots (resource) from a barrier ID when a task is scheduled by the main list
// scheduler
BarrierResourceState::barrierType BarrierResourceState::assignBarrierSlots(
        producerSlotsType slotDemand, mlir::Operation* op,
        std::map<mlir::Operation*, SmallVector<mlir::Operation*>>& taskConsumerMap) {
    availableSlotKey key(slotDemand);
    availableSlotsIteratorType itr = _availableProducerSlots.lower_bound(key);

    const auto isSameConsumerVectors = [&](SmallVector<mlir::Operation*> a, SmallVector<mlir::Operation*> b) {
        for (const auto& task : a) {
            if (std::find(b.begin(), b.end(), task) == b.end())
                return false;
        }

        for (const auto& task : b) {
            if (std::find(a.begin(), a.end(), task) == a.end())
                return false;
        }

        return true;
    };

    {
        availableSlotsIteratorType retItr = itr;
        VPUX_UNUSED(taskConsumerMap);
        for (; itr != _availableProducerSlots.end(); ++itr) {
            if (itr->isBarrierInUse()) {
                barrierType currentBid = itr->_barrier;
                auto users = _barrierUsers[currentBid - 1UL];
                VPUX_THROW_UNLESS(!users.empty(), "Barrier is not used");
                for (auto& user : users) {
                    if (isSameConsumerVectors(taskConsumerMap[user], taskConsumerMap[op])) {
                        if (assignBarrierSlots(currentBid, slotDemand)) {
                            _barrierUsers[currentBid - 1UL].insert(op);
                            return currentBid;
                        } else
                            break;
                    }
                }
            }
        }

        itr = retItr;
        for (; (itr != _availableProducerSlots.end()) && (itr->isBarrierInUse()); ++itr) {
        }
        if (itr != _availableProducerSlots.end()) {
            retItr = itr;
        };
        itr = retItr;
    }

    if ((itr == _availableProducerSlots.end()) || (itr->_availableProducerSlots < slotDemand)) {
        return invalidBarrier();
    }

    barrierType barrierId = itr->_barrier;
    if (assignBarrierSlots(barrierId, slotDemand)) {
        _barrierUsers[barrierId - 1UL].insert(op);
        return barrierId;
    } else {
        return invalidBarrier();
    }
}

// Assigns the requested producers slots (resource) from a barrier ID when a task is scheduled by the main list
// scheduler
bool BarrierResourceState::assignBarrierSlots(barrierType barrierId, producerSlotsType slotDemand) {
    VPUX_THROW_UNLESS((barrierId <= _barrierReference.size()) && (barrierId >= 1UL), "Invalid barrierId supplied");
    availableSlotsIteratorType itr = _barrierReference[barrierId - 1UL];

    VPUX_THROW_UNLESS((itr->_availableProducerSlots) >= slotDemand,
                      "Error _availableProducerSlots {0} >=slotDemand {1}", itr->_availableProducerSlots, slotDemand);
    producerSlotsType newSlotDemand = (itr->_availableProducerSlots) - slotDemand;

    itr = update(itr, newSlotDemand);
    return (itr != _availableProducerSlots.end());
}

// Releases the producers slots (resource) from a barrier ID when a task is unscheduled by the main list scheduler
bool BarrierResourceState::unassignBarrierSlots(barrierType barrierId, producerSlotsType slotDemand,
                                                mlir::Operation* op) {
    VPUX_THROW_UNLESS((barrierId <= _barrierReference.size()) && (barrierId >= 1UL), "Invalid barrierId supplied");
    availableSlotsIteratorType itr = _barrierReference[barrierId - 1UL];
    producerSlotsType newSlotDemand = (itr->_availableProducerSlots) + slotDemand;

    itr = update(itr, newSlotDemand);
    _barrierUsers[barrierId - 1UL].erase(op);
    return (itr != _availableProducerSlots.end());
}

BarrierResourceState::barrierType BarrierResourceState::invalidBarrier() {
    return std::numeric_limits<barrierType>::max();
}

// NOTE: will also update _barrierReference
void BarrierResourceState::update(barrierType barrierId, producerSlotsType newSlotsValue) {
    VPUX_THROW_UNLESS((barrierId <= _barrierReference.size()) && (barrierId >= 1UL), "Invalid barrierId supplied");
    availableSlotsIteratorType itr = _barrierReference[barrierId - 1UL];
    update(itr, newSlotsValue);
}

// Updates the state of a barrier with current available producers slots
// This should be called whenever a task is scheduled/unscheduled by the main list scheduler
BarrierResourceState::availableSlotsIteratorType BarrierResourceState::update(availableSlotsIteratorType itr,
                                                                              producerSlotsType newSlotsValue) {
    VPUX_THROW_UNLESS(itr != _availableProducerSlots.end(), "Invalid _availableProducerSlots iterator");

    availableSlotKey key = *itr;
    key._availableProducerSlots = newSlotsValue;
    _availableProducerSlots.erase(itr);

    itr = (_availableProducerSlots.insert(key)).first;
    VPUX_THROW_UNLESS(itr != _availableProducerSlots.end(), "Invalid _availableProducerSlots iterator");

    _barrierReference[(itr->_barrier) - 1UL] = itr;
    return itr;
}
