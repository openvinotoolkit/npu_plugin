//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/barrier_scheduler.hpp"
#include "vpux/compiler/dialect/VPURT/cycle_based_barrier_scheduler.hpp"
#include "vpux/compiler/dialect/VPURT/passes.hpp"

#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/DenseMap.h>

using namespace vpux;

namespace {
class AssignVirtualBarriersPass final : public VPURT::AssignVirtualBarriersBase<AssignVirtualBarriersPass> {
public:
    explicit AssignVirtualBarriersPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AssignVirtualBarriersPass::safeRunOnFunc() {
    auto func = getFunction();

    auto cycleBasedBarrierScheduler =
            useCycleBasedBarrierScheduler.hasValue() ? useCycleBasedBarrierScheduler.getValue() : true;
    auto numBarriersToUse = numBarriers.hasValue() ? numBarriers.getValue() : VPUIP::getNumAvailableBarriers(func);
    static constexpr int64_t MAX_PRODUCER_SLOT_COUNT = 256;
    auto numSlotsPerBarrierToUse =
            numSlotsPerBarrier.hasValue() ? numSlotsPerBarrier.getValue() : MAX_PRODUCER_SLOT_COUNT;
    _log.trace("There are {0} Barriers to use", numBarriersToUse);
    _log.trace("There are {0} slots for each barrier", numSlotsPerBarrierToUse);

    // A task can only start on the runtime when the following conditions are true.
    // (1) All the task’s wait barriers have zero producers.
    // (2) All the task’s update barriers are ready (the physical barrier register has been programmed by the LeonNN as
    // new virtual barrier).
    // A barrier will only be reset (configured to be another virtual barrier) when its consumer count is zero.

    if (cycleBasedBarrierScheduler) {
        VPURT::CycleBasedBarrierScheduler barrierScheduler(func, _log);
        barrierScheduler.init(numBarriersToUse, numSlotsPerBarrierToUse);

        barrierScheduler.generateScheduleWithBarriers();
        bool success = barrierScheduler.performRuntimeSimulation();
        if (!success) {
            VPUX_THROW("Barrier scheduling and/or runtime simulation was not suceessful");
        }

        barrierScheduler.clearTemporaryAttributes();
    } else {
        // The reason for this loop is explained on E#28923
        // The barrier scheduler in its current form does not model the consumers count of a barrier. It is for this
        // reason,
        // when runtime simulation is performed to assign physical barrier ID's to the virtual barriers, it may fail to
        // perform a valid assignment due to the fact that the barrier scheduler may have overallocated active barriers
        // during scheduling.

        // The current solution to this is to reduce the number of barrier available to the scheduler and re-preform the
        // scheduling. It should be possible to schedule any graph with a minimum of two barriers This condition can be
        // removed when the compiler transitions to using a defined task execution order earlier in compilation and the
        // memory scheduler guarantees that the maximum number of active barrier (parallel tasks) will not exceed the
        // limit. Such a feature will significantly simply barrier allocation and would be a prerequisite for moving
        // 'barrier safety' from the runtime to the compiler. The definition of barrier safety is that it can be
        // guaranteed the barriers will be reprogrammed by the LeonNN during inference at the correct time during an
        // inference.

        VPURT::BarrierScheduler barrierScheduler(func, _log);
        barrierScheduler.init();

        bool success = false;
        for (size_t barrier_bound = (numBarriersToUse / 2); !success && (barrier_bound >= 1UL); --barrier_bound) {
            barrierScheduler.generateScheduleWithBarriers(barrier_bound, numSlotsPerBarrierToUse);
            success = barrierScheduler.performRuntimeSimulation();
        }
        barrierScheduler.clearTemporaryAttributes();

        if (!success) {
            VPUX_THROW("Barrier scheduling and/or runtime simulation was not suceessful");
        }
    }
}

}  // namespace

//
// createAssignVirtualBarriersPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createAssignVirtualBarriersPass(Logger log) {
    return std::make_unique<AssignVirtualBarriersPass>(log);
}
