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
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/barrier_resource_state.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#pragma once

namespace vpux {

static constexpr StringLiteral uniqueIdAttrName = "uniqueId";
static constexpr StringLiteral virtualIdAttrName = "VPURT.virtualId";

typedef mlir::Operation const* operation_t;

struct barrier_info_t {
    barrier_info_t(size_t bindex = 0UL, size_t slot_count = 0UL): bindex_(bindex), slot_count_(slot_count) {
    }
    size_t bindex_;
    size_t slot_count_;
}; /*struct barrier_info_t*/

typedef std::unordered_map<operation_t, barrier_info_t> active_barrier_map_t;
// typedef active_barrier_map_t::iterator barrier_map_iterator_t;
typedef size_t resource_t;

struct op_resource_state_t {
    op_resource_state_t(size_t n = 0UL, size_t m = 0UL)
            : barrier_map_(), state_(), barrier_count_(n), slots_per_barrier_(m) {
        Logger::global().error("Initializing op_resource_state in Barrier_Schedule_Generator with barrier count {0} "
                               "slots_per_barrie {1}",
                               barrier_count_, slots_per_barrier_);
    }

    void init(const op_resource_state_t& other) {
        barrier_map_.clear();
        barrier_count_ = other.barrier_count_;
        slots_per_barrier_ = other.slots_per_barrier_;
        state_.init(barrier_count_, slots_per_barrier_);
    }

    bool is_resource_available(const resource_t& demand) const {
        Logger::global().error("Looking for a barrier with free slots");
        return state_.has_barrier_with_slots(demand);
    }

    bool schedule_operation(const operation_t& op, resource_t& demand) {
        Logger::global().error("Scheduling an operation");
        assert(is_resource_available(demand));
        if (barrier_map_.find(op) != barrier_map_.end()) {
            // Logger::global().error("Could not find operation  {0} in the barrier map" ,
            // getUniqueID(const_cast<mlir::Operation*>(op)), demand);
            return false;
        }
        // Logger::global().error("Inserting operation  {0}, in barrier_map_ with demand {1}" ,
        // getUniqueID(const_cast<mlir::Operation*>(op)), demand);
        size_t bid = state_.assign_slots(demand);
        barrier_map_.insert(std::make_pair(op, barrier_info_t(bid, demand)));
        return true;
    }

    bool unschedule_operation(const operation_t& op) {
        auto itr = barrier_map_.find(op);
        if (itr == barrier_map_.end()) {
            return false;
        }
        const barrier_info_t& binfo = itr->second;
        bool ret = state_.unassign_slots(binfo.bindex_, binfo.slot_count_);
        assert(ret);
        (void)ret;
        barrier_map_.erase(itr);
        return true;
    }

    mlir::IntegerAttr getUniqueID(const mlir::Operation* op) const {
        auto taskOp = mlir::dyn_cast<VPUIP::TaskOpInterface>(const_cast<mlir::Operation*>(op));
        return taskOp->getAttr(uniqueIdAttrName).dyn_cast_or_null<mlir::IntegerAttr>();
    }

    const barrier_info_t& get_barrier_info(const operation_t& op) const {
        // Logger::global().error("Looking for operation {0}, in barrier_map_", getUniqueID(op));
        auto itr = barrier_map_.find(op);

        assert(itr != barrier_map_.end());
        return itr->second;
    }

    active_barrier_map_t barrier_map_;
    BarrierResourceState state_;
    size_t barrier_count_;
    size_t slots_per_barrier_;
}; /*struct op_resource_state_t*/

typedef op_resource_state_t resource_state_t;

typedef size_t schedule_time_t;
struct schedule_info_t {
    schedule_time_t schedule_time_;
    operation_t op_;
    size_t barrier_index_;
    size_t slot_count_;
}; /*struct schedule_info_t*/

struct barrier_scheduler_traits {
    // typedef op_resource_state_t resource_state_t;

    static void initialize_resource_state(const resource_state_t& start_state, resource_state_t& state) {
        state.init(start_state);
    }
};

}  // namespace vpux
