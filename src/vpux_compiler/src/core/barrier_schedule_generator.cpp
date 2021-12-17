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

#include "vpux/compiler/core/barrier_schedule_generator.hpp"

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/range.hpp"

using namespace vpux;

//
// Constructor
//

BarrierScheduleGenerator::BarrierScheduleGenerator(
        mlir::MLIRContext* ctx, mlir::FuncOp func, size_t n, size_t m,
        std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>,
                 task_operation_comparator_by_schedule_time_t>& taskOpUpdateWaitMap)
        : barrierCount_(n),
          slotsPerBarrier_(m),
          startState_(n, m),
          scheduler_begin_(ctx, func, startState_, taskOpUpdateWaitMap),
          scheduler_end_(ctx, func),
          sinfo_() {
}

BarrierScheduleGenerator::BarrierScheduleGenerator(mlir::MLIRContext* ctx, mlir::FuncOp func)
        : barrierCount_(0UL),
          slotsPerBarrier_(0UL),
          startState_(),
          scheduler_begin_(ctx, func),
          scheduler_end_(ctx, func){};

bool BarrierScheduleGenerator::reached_end() const {
    return scheduler_begin_ == scheduler_end_;
}

bool BarrierScheduleGenerator::operator==(const BarrierScheduleGenerator& o) const {
    return reached_end() && o.reached_end();
}

bool BarrierScheduleGenerator::operator!=(const BarrierScheduleGenerator& o) const {
    return !(*this == o);
}

void BarrierScheduleGenerator::operator++() {
    ++scheduler_begin_;
}

const BarrierScheduleGenerator::schedule_info_t& BarrierScheduleGenerator::operator*(void) {
    sinfo_.op_ = *scheduler_begin_;
    sinfo_.schedule_time_ = scheduler_begin_.current_time();
    const resource_state_t& rstate = scheduler_begin_.resource_state();
    Logger::global().error("Get barrier info for operation {0}", FeasibleScheduleGenerator::getUniqueID(sinfo_.op_));
    const barrier_info_t& binfo = rstate.get_barrier_info(sinfo_.op_);
    sinfo_.barrier_index_ = binfo.bindex_;
    sinfo_.slot_count_ = binfo.slot_count_;
    return sinfo_;
}
