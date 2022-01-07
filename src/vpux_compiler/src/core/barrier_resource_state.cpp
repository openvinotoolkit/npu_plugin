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
// Constructor
//

BarrierResourceState::BarrierResourceState(): _barrierReference(), _availableProducerSlots() {
}

BarrierResourceState::BarrierResourceState(size_t barrierCount, size_t producerSlotCount)
        : _barrierReference(), _availableProducerSlots() {
    init(barrierCount, producerSlotCount);
}

void BarrierResourceState::init(const size_t barrierCount, const producerSlotsType producerSlotCount) {
    VPUX_THROW_UNLESS((barrierCount && producerSlotCount), "Error number of barrier and/or producer slot count is 0");
    _availableProducerSlots.clear();
    _barrierReference.clear();

    availableSlotsIteratorType hint = _availableProducerSlots.end();
    for (size_t barrierId = 1UL; barrierId <= barrierCount; barrierId++) {
        hint = _availableProducerSlots.insert(hint, availableSlotKey(producerSlotCount, barrierType(barrierId)));
        _barrierReference.push_back(hint);
    }
}

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

// Precondition: hasBarrierWithSlots(slotDemand) is true //
BarrierResourceState::barrierType BarrierResourceState::assignBarrierSlots(producerSlotsType slotDemand) {
    availableSlotKey key(slotDemand);
    availableSlotsIteratorType itr = _availableProducerSlots.lower_bound(key);
    {
        availableSlotsIteratorType retItr = itr;

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
    return assignBarrierSlots(barrierId, slotDemand) ? barrierId : invalidBarrier();
}

bool BarrierResourceState::assignBarrierSlots(barrierType barrierId, producerSlotsType slotDemand) {

    VPUX_THROW_UNLESS((barrierId <= _barrierReference.size()) && (barrierId >= 1UL), "Invalid barrierId supplied");
    availableSlotsIteratorType itr = _barrierReference[barrierId - 1UL];

    VPUX_THROW_UNLESS((itr->_availableProducerSlots) >= slotDemand, "Error _availableProducerSlots {0} >=slotDemand {1}", itr->_availableProducerSlots, slotDemand);
    producerSlotsType newSlotDemand = (itr->_availableProducerSlots) - slotDemand;

    itr = update(itr, newSlotDemand);
    return (itr != _availableProducerSlots.end());
}

bool BarrierResourceState::unassignBarrierSlots(barrierType barrierId, producerSlotsType slotDemand) {
    VPUX_THROW_UNLESS((barrierId <= _barrierReference.size()) && (barrierId >= 1UL), "Invalid barrierId supplied");
    availableSlotsIteratorType itr = _barrierReference[barrierId - 1UL];
    producerSlotsType newSlotDemand = (itr->_availableProducerSlots) + slotDemand;

    itr = update(itr, newSlotDemand);
    return (itr != _availableProducerSlots.end());
}

BarrierResourceState::barrierType BarrierResourceState::invalidBarrier() {
    return std::numeric_limits<barrierType>::max();
}

// NOTE: will also update _barrierReference //
void BarrierResourceState::update(barrierType barrierId, producerSlotsType newSlotsValue) {
    VPUX_THROW_UNLESS((barrierId <= _barrierReference.size()) && (barrierId >= 1UL), "Invalid barrierId supplied");
    availableSlotsIteratorType itr = _barrierReference[barrierId - 1UL];
    update(itr, newSlotsValue);
}

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