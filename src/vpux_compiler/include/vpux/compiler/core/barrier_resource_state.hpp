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

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/DenseSet.h>

namespace vpux {

namespace VPURT {

class BarrierResourceState {
public:
    BarrierResourceState();
    BarrierResourceState(size_t barrier_count, size_t slot_count);

    using slots_t = size_t;
    using barrier_t = size_t;

    struct available_slot_key_t {
        available_slot_key_t(slots_t slots = slots_t(0UL), barrier_t barrier = barrier_t(0UL))
                : available_slots_(slots), total_slots_(slots), barrier_(barrier) {
        }

        bool operator<(const available_slot_key_t& o) const {
            return (o.available_slots_ != available_slots_) ? (available_slots_ < o.available_slots_)
                                                            : (barrier_ < o.barrier_);
        }

        bool barrier_in_use() const {
            return total_slots_ > available_slots_;
        }

        slots_t available_slots_;
        const slots_t total_slots_;
        barrier_t barrier_;
    };  // struct available_slot_key_t //

    using available_slots_t = std::set<available_slot_key_t>;
    using const_available_slots_iterator_t = typename available_slots_t::const_iterator;
    using available_slots_iterator_t = typename available_slots_t::iterator;
    using barrier_reference_t = std::vector<available_slots_iterator_t>;

    void init(size_t bcount, slots_t slot_count);
    bool has_barrier_with_slots(slots_t slot_demand) const;
    barrier_t assign_slots(slots_t slot_demand);
    bool assign_slots(barrier_t bid, slots_t slot_demand);
    bool unassignSlots(barrier_t bid, slots_t slot_demand);
    static barrier_t invalid_barrier();
    void update(barrier_t bid, slots_t new_slots_value);
    available_slots_iterator_t update(available_slots_iterator_t itr, slots_t new_slots_value);

private:
    barrier_reference_t barrier_reference_;
    available_slots_t available_slots_;
};

}  // namespace VPURT
}  // namespace vpux
