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

// #include "vpux/compiler/core/barrier_schedule_generator.hpp"

// #include "vpux/compiler/utils/attributes.hpp"

// #include "vpux/utils/core/range.hpp"

// using namespace vpux;

// //
// // Constructor
// //

// BarrierScheduleGenerator::BarrierScheduleGenerator(Logger log, mlir::MLIRContext* ctx, mlir::FuncOp func, size_t n, size_t m = 1UL)
//         : barrierCount_(n),
//           slotsPerBarrier_(m),
//           startState_(n, m),
//           scheduler_begin_(ctx, func, startState_, log),
//           scheduler_end_(ctx, func, log),
//           sinfo_(),
//           _log(log)
//            {
// }

// BarrierScheduleGenerator::BarrierScheduleGenerator(Logger log, mlir::MLIRContext* ctx, mlir::FuncOp func)
//         : barrierCount_(0UL),
//           slotsPerBarrier_(0UL),
//           startState_(),
//           scheduler_begin_(ctx, func, log),
//           scheduler_end_(ctx, func, log),
//           _log(log)
//           {};

// bool BarrierScheduleGenerator::reached_end() const {
//     return scheduler_begin_ == scheduler_end_;
// }

// bool BarrierScheduleGenerator::operator==(const BarrierScheduleGenerator& o) const {
//     return reached_end() && o.reached_end();
// }

// bool BarrierScheduleGenerator::operator!=(const BarrierScheduleGenerator& o) const {
//     return !(*this == o);
// }

// void BarrierScheduleGenerator::operator++() {
//     ++scheduler_begin_;
// }

// const BarrierScheduleGenerator::schedule_info_t& BarrierScheduleGenerator::operator*(void) {
//     sinfo_.op_ = *scheduler_begin_;
//     sinfo_.schedule_time_ = scheduler_begin_.currentTime();
//     const resource_state_t& rstate = scheduler_begin_.resourceState();
//     Logger::global().error("Get barrier info for operation {0}", FeasibleBarrierScheduler::getUniqueID(sinfo_.op_));
//     const barrier_info_t& binfo = rstate.get_barrier_info(sinfo_.op_);
//     sinfo_.barrier_index_ = binfo.bindex_;
//     sinfo_.slot_count_ = binfo.slot_count_;
//     return sinfo_;
// }
