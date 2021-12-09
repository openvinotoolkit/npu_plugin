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

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/range.hpp"

using namespace vpux;

//
// Constructor
//

BarrierResourceState::BarrierResourceState(): barrier_reference_(), available_slots_() {
}

BarrierResourceState::BarrierResourceState(size_t barrier_count, size_t slot_count)
        : barrier_reference_(), available_slots_() {
    Logger::global().error("Initializing Barrier_Resource_State");
    init(barrier_count, slot_count);
}

void BarrierResourceState::init(size_t bcount, slots_t slot_count) {
    assert(bcount && slot_count);
    available_slots_.clear();
    barrier_reference_.clear();

    // amortized O(1) insert cost //
    available_slots_iterator_t hint = available_slots_.end();
    for (size_t bid = 1UL; bid <= bcount; bid++) {
        hint = available_slots_.insert(hint, available_slot_key_t(slot_count, barrier_t(bid)));
        barrier_reference_.push_back(hint);
    }
}

bool BarrierResourceState::has_barrier_with_slots(slots_t slot_demand) const {
    Logger::global().error("Checking if there is a barrier free with {0} free slots", slot_demand);
    available_slot_key_t key(slot_demand);
    const_available_slots_iterator_t itr = available_slots_.lower_bound(key);
    const_available_slots_iterator_t ret_itr = itr;

    // prefer a unused barrier to satisfy this slot demand //
    for (; (itr != available_slots_.end()) && (itr->barrier_in_use()); ++itr) {
    }
    if (itr != available_slots_.end()) {
        ret_itr = itr;
    };

    bool result = ret_itr != available_slots_.end();
    Logger::global().error("There is True/False {0} a barrier with free slots", result);
    return ret_itr != available_slots_.end();
}

// Precondition: has_barrier_with_slots(slot_demand) is true //
BarrierResourceState::barrier_t BarrierResourceState::assign_slots(slots_t slot_demand) {
    available_slot_key_t key(slot_demand);
    available_slots_iterator_t itr = available_slots_.lower_bound(key);
    {
        available_slots_iterator_t ret_itr = itr;

        for (; (itr != available_slots_.end()) && (itr->barrier_in_use()); ++itr) {
        }
        if (itr != available_slots_.end()) {
            ret_itr = itr;
        };
        itr = ret_itr;
    }

    if ((itr == available_slots_.end()) || (itr->available_slots_ < slot_demand)) {
        return invalid_barrier();
    }

    barrier_t bid = itr->barrier_;
    return assign_slots(bid, slot_demand) ? bid : invalid_barrier();
}

bool BarrierResourceState::assign_slots(barrier_t bid, slots_t slot_demand) {
    assert((bid <= barrier_reference_.size()) && (bid >= 1UL));
    available_slots_iterator_t itr = barrier_reference_[bid - 1UL];
    assert((itr->available_slots_) >= slot_demand);
    slots_t new_slot_demand = (itr->available_slots_) - slot_demand;

    itr = update(itr, new_slot_demand);
    return (itr != available_slots_.end());
}

bool BarrierResourceState::unassign_slots(barrier_t bid, slots_t slot_demand) {
    assert((bid <= barrier_reference_.size()) && (bid >= 1UL));
    available_slots_iterator_t itr = barrier_reference_[bid - 1UL];
    slots_t new_slot_demand = (itr->available_slots_) + slot_demand;

    itr = update(itr, new_slot_demand);
    return (itr != available_slots_.end());
}

BarrierResourceState::barrier_t BarrierResourceState::invalid_barrier() {
    return std::numeric_limits<barrier_t>::max();
}

// NOTE: will also update barrier_reference_ //
void BarrierResourceState::update(barrier_t bid, slots_t new_slots_value) {
    assert((bid <= barrier_reference_.size()) && (bid >= 1UL));
    available_slots_iterator_t itr = barrier_reference_[bid - 1UL];
    update(itr, new_slots_value);
}

BarrierResourceState::available_slots_iterator_t BarrierResourceState::update(available_slots_iterator_t itr,
                                                                              slots_t new_slots_value) {
    assert(itr != available_slots_.end());

    available_slot_key_t key = *itr;
    key.available_slots_ = new_slots_value;
    available_slots_.erase(itr);

    itr = (available_slots_.insert(key)).first;
    assert(itr != available_slots_.end());
    barrier_reference_[(itr->barrier_) - 1UL] = itr;
    return itr;
}