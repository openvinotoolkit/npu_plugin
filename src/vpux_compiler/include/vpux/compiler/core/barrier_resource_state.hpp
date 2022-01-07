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

#pragma once

#include "vpux/utils/core/error.hpp"
namespace vpux {

namespace VPURT {

class BarrierResourceState final {
public:
    BarrierResourceState();
    BarrierResourceState(size_t barrierCount, size_t producerSlotCount);

    using producerSlotsType = size_t;
    using barrierType = size_t;

    struct availableSlotKey {
        availableSlotKey(producerSlotsType slots = producerSlotsType(0UL), barrierType barrier = barrierType(0UL))
                : _availableProducerSlots(slots), _totalProducerSlots(slots), _barrier(barrier) {
        }

        bool operator<(const availableSlotKey& o) const {
            return (o._availableProducerSlots != _availableProducerSlots)
                           ? (_availableProducerSlots < o._availableProducerSlots)
                           : (_barrier < o._barrier);
        }

        bool isBarrierInUse() const {
            return _totalProducerSlots > _availableProducerSlots;
        }

        producerSlotsType _availableProducerSlots;
        const producerSlotsType _totalProducerSlots;
        barrierType _barrier;
    };  // struct availableSlotKey //

    using availableProducerSlotsType = std::set<availableSlotKey>;
    using constAvailableslotsIteratorType = typename availableProducerSlotsType::const_iterator;
    using availableSlotsIteratorType = typename availableProducerSlotsType::iterator;
    using barrierReferenceType = std::vector<availableSlotsIteratorType>;

    void init(const size_t barrierCount, const producerSlotsType producerSlotCount);
    bool hasBarrierWithSlots(producerSlotsType slotDemand) const;
    barrierType assignBarrierSlots(producerSlotsType slotDemand);
    bool assignBarrierSlots(barrierType barrierId, producerSlotsType slotDemand);
    bool unassignBarrierSlots(barrierType barrierId, producerSlotsType slotDemand);
    static barrierType invalidBarrier();
    void update(barrierType barrierId, producerSlotsType newSlotsValue);
    availableSlotsIteratorType update(availableSlotsIteratorType itr, producerSlotsType newSlotsValue);

private:
    barrierReferenceType _barrierReference;
    availableProducerSlotsType _availableProducerSlots;
};

}  // namespace VPURT
}  // namespace vpux
