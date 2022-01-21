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
    BarrierResourceState(size_t barrierCount, size_t maximumProducerSlotCount);

    // Container that describes for each barrier ID
    // (1) The barriers total allowable producer number i.e. 256
    // (2) The current available producer slots i.e (256 - used slots)
    // (3) If the barrier is currently in use i.e its total slots != current available slots
    struct availableSlotKey {
        availableSlotKey(size_t slots = size_t(0UL), size_t barrier = size_t(0UL))
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

        size_t _availableProducerSlots;
        const size_t _totalProducerSlots;
        size_t _barrier;
    };  // struct availableSlotKey

    using availableProducerSlotsType = std::set<availableSlotKey>;
    using constAvailableslotsIteratorType = typename availableProducerSlotsType::const_iterator;
    using availableSlotsIteratorType = typename availableProducerSlotsType::iterator;
    using barrierReferenceType = std::vector<availableSlotsIteratorType>;

    void init(const size_t barrierCount, const size_t maximumProducerSlotCount);
    bool hasBarrierWithAvailableSlots(size_t slotDemand);
    constAvailableslotsIteratorType findBarrierWithAvailableSlots(size_t slotDemand);
    size_t assignBarrierSlots(size_t slotDemand);
    bool assignBarrierSlots(size_t barrierId, size_t slotDemand);
    bool unassignBarrierSlots(size_t barrierId, size_t slotDemand);
    static size_t invalidBarrier();
    void update(size_t barrierId, size_t updatedAvailableProducerSlots);
    availableSlotsIteratorType update(availableSlotsIteratorType itr, size_t updatedAvailableProducerSlots);

private:
    // A vector of iterators to each entry in the _availableProducerSlots container
    barrierReferenceType _barrierReference;
    // Stores an availableSlotKey struct for each barrier.
    // i.e. Information for each barrier, its total available slots (256) and the current free slots
    availableProducerSlotsType _globalAvailableProducerSlots;
};

}  // namespace VPURT
}  // namespace vpux
