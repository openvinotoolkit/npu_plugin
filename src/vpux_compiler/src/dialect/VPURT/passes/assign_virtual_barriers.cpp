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

#include "vpux/compiler/core/feasible_barrier_generator.hpp"

#include "vpux/compiler/dialect/VPUIP/utils.hpp"
#include "vpux/compiler/dialect/VPURT/passes.hpp"

#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/DenseMap.h>

using namespace vpux;

namespace {

// Same value for all architectures for now
constexpr int64_t MAX_BARRIERS_FOR_ARCH = 64;

class AssignVirtualBarriersPass final : public VPURT::AssignVirtualBarriersBase<AssignVirtualBarriersPass> {
public:
    explicit AssignVirtualBarriersPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AssignVirtualBarriersPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    auto numBarriersToUse = numBarriers.hasValue() ? numBarriers.getValue() : VPUIP::getNumAvailableBarriers(func);
    static constexpr int64_t MAX_SLOT_COUNT = 256;
    auto numSlotsPerBarrierToUse = numSlotsPerBarrier.hasValue() ? numSlotsPerBarrier.getValue() : MAX_SLOT_COUNT;

    VPURT::FeasibleBarrierScheduler barrierScheduler(&ctx, func, _log);
    barrierScheduler.init();

    bool success = false;
    for (size_t barrier_bound = (numBarriersToUse / 2); !success && (barrier_bound >= 1UL); --barrier_bound) {
        barrierScheduler.schedule(barrier_bound, numSlotsPerBarrierToUse);
        success = barrierScheduler.performRuntimeSimulation();
    }
    // barrierScheduler.clearUniqueID();

    if (!success) {
        VPUX_THROW("Barrier scheduling and/or runtime simulation was not suceessful");
    }
}

}  // namespace

//
// createAssignVirtualBarriersPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createAssignVirtualBarriersPass(Logger log) {
    return std::make_unique<AssignVirtualBarriersPass>(log);
}
