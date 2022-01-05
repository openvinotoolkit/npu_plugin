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
// #include "vpux/compiler/core/attributes/shape.hpp"
// #include "vpux/compiler/core/barrier_resource_state.hpp"
// #include "vpux/compiler/dialect/IERT/ops.hpp"
// #include "vpux/compiler/dialect/VPUIP/ops.hpp"

// #pragma once

// namespace vpux {

// namespace VPURT {

// static constexpr StringLiteral uniqueIdAttrName = "uniqueId";
// static constexpr StringLiteral virtualIdAttrName = "VPURT.virtualId";

// struct barrier_info_t {
//     barrier_info_t(size_t bindex = 0UL, size_t slot_count = 0UL): bindex_(bindex), slot_count_(slot_count) {
//     }
//     size_t bindex_;
//     size_t slot_count_;
// };

// using operation_t = mlir::Operation*;
// using active_barrier_map_t = std::unordered_map<operation_t, barrier_info_t>;
// using resource_t = size_t;

// struct op_resource_state_t {
//     op_resource_state_t(size_t barrierCount = 0UL, size_t slotsPerBarrier = 0UL)
//             : _barrierMap(), _state(), _barrierCount(barrierCount), _slotsPerBarrier(slotsPerBarrier) {
//         Logger::global().error("Initializing op_resource_state in Barrier_Schedule_Generator with barrier count {0} "
//                                "slots_per_barrie {1}",
//                                _barrierCount, _slotsPerBarrier);
//     }

//     void init(const op_resource_state_t& other) {
//         _barrierMap.clear();
//         _barrierCount = other._barrierCount;
//         _slotsPerBarrier = other._slotsPerBarrier;
//         _state.init(_barrierCount, _slotsPerBarrier);
//     }

//     bool is_resource_available(const resource_t& producerSlotRequirement) const {
//         Logger::global().error("Looking for a barrier with free slots");
//         return _state.has_barrier_with_slots(producerSlotRequirement);
//     }

//     bool schedule_operation(const operation_t& op, resource_t& producerSlotRequirement) {
//         Logger::global().error("Scheduling an operation");
//         assert(is_resource_available(producerSlotRequirement));
//         if (_barrierMap.find(op) != _barrierMap.end()) {
//             return false;
//         }
//         size_t bid = _state.assign_slots(producerSlotRequirement);
//         _barrierMap.insert(std::make_pair(op, barrier_info_t(bid, producerSlotRequirement)));
//         return true;
//     }

//     bool unschedule_operation(const operation_t& op) {
//         auto itr = _barrierMap.find(op);
//         if (itr == _barrierMap.end()) {
//             return false;
//         }
//         const barrier_info_t& binfo = itr->second;
//         bool ret = _state.unassign_slots(binfo.bindex_, binfo.slot_count_);
//         assert(ret);
//         (void)ret;
//         _barrierMap.erase(itr);
//         return true;
//     }

//     mlir::IntegerAttr getUniqueID(const mlir::Operation* op) const {
//         auto taskOp = mlir::dyn_cast<VPUIP::TaskOpInterface>(const_cast<mlir::Operation*>(op));
//         return taskOp->getAttr(uniqueIdAttrName).dyn_cast_or_null<mlir::IntegerAttr>();
//     }

//     const barrier_info_t& get_barrier_info(const operation_t& op) const {
//         auto itr = _barrierMap.find(op);

//         assert(itr != _barrierMap.end());
//         return itr->second;
//     }

//     active_barrier_map_t _barrierMap;
//     BarrierResourceState _state;
//     size_t _barrierCount;
//     size_t _slotsPerBarrier;
// }; /*struct op_resource_state_t*/

// using resource_state_t = op_resource_state_t;

// using schedule_time_t = size_t;

//}  // namespace VPURT
//}  // namespace vpux
