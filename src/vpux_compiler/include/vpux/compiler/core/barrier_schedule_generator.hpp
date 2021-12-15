// //
// // Copyright Intel Corporation.
// //
// // LEGAL NOTICE: Your use of this software and any required dependent software
// // (the "Software Package") is subject to the terms and conditions of
// // the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// // which may also include notices, disclaimers, or license terms for
// // third party or open source software included in or with the Software Package,
// // and your use indicates your acceptance of all such terms. Please refer
// // to the "third-party-programs.txt" or other similarly-named text file
// // included with the Software Package for additional details.
// //

// #pragma once
// #include "vpux/compiler/core/feasible_schedule_generator.hpp"

// #include "vpux/utils/core/func_ref.hpp"
// #include "vpux/utils/core/logger.hpp"
// #include "vpux/utils/core/small_vector.hpp"

// #include "vpux/compiler/core/attributes/strides.hpp"
// #include "vpux/compiler/dialect/IE/ops.hpp"
// #include "vpux/compiler/dialect/IERT/ops.hpp"
// #include "vpux/compiler/utils/attributes.hpp"
// #include "vpux/compiler/utils/error.hpp"

// #include <mlir/Dialect/Async/IR/Async.h>
// #include <mlir/IR/BuiltinOps.h>
// #include <mlir/IR/Operation.h>

// #include "vpux/utils/core/checked_cast.hpp"
// #include "vpux/utils/core/error.hpp"
// #include "vpux/utils/core/format.hpp"
// #include "vpux/utils/core/numeric.hpp"

// #include <mlir/Dialect/MemRef/IR/MemRef.h>
// #include <mlir/Dialect/StandardOps/IR/Ops.h>
// #include <mlir/IR/Value.h>
// #include <mlir/Transforms/DialectConversion.h>

// #include <llvm/ADT/BitVector.h>
// #include <llvm/ADT/DenseSet.h>

// #include "vpux/compiler/core/barrier_resource_state.hpp"

// namespace vpux {

// // Forward declaration
// class FeasibleBarrierScheduler;
// class BarrierScheduleGenerator {
// public:
//     BarrierScheduleGenerator(Logger log, mlir::MLIRContext* ctx, mlir::FuncOp func, size_t n, size_t m);
//     BarrierScheduleGenerator(Logger log, mlir::MLIRContext* ctx, mlir::FuncOp func);

//     typedef size_t schedule_time_t;
//     struct schedule_info_t {
//         schedule_time_t schedule_time_;
//         mlir::Operation* op_;
//         size_t barrier_index_;
//         size_t slot_count_;
//     }; /*struct schedule_info_t*/

//     bool operator==(const BarrierScheduleGenerator& o) const;
//     bool operator!=(const BarrierScheduleGenerator& o) const;
//     void operator++();
//     const schedule_info_t& operator*(void);
//     bool reached_end() const;

//     struct barrier_scheduler_traits {

//         static void initialize_resource_state(const resource_state_t& start_state, resource_state_t& state) {
//             state.init(start_state);
//         }

//         static bool is_resource_available(const resource_t& demand, const resource_state_t& state) {
//             return state.is_resource_available(demand);
//         }

//         static bool schedule_operation(mlir::Operation*& op, resource_t demand, resource_state_t& state) {
//             return state.schedule_operation(op, demand);
//         }

   

//         static bool unschedule_operation(mlir::Operation*& op, resource_state_t& rstate) {
//             return rstate.unschedule_operation(op);
//         }

//     };  // struct barrier_scheduler_traits //

// private:
//     Logger _log;
//     size_t barrierCount_;
//     size_t slotsPerBarrier_;
//     const resource_state_t startState_;
//     FeasibleBarrierScheduler scheduler_begin_;
//     FeasibleBarrierScheduler scheduler_end_;
//     mutable schedule_info_t sinfo_;

// };  // class Barrier_Schedule_Generator //

// }  // namespace vpux
