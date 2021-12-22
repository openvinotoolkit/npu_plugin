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

#include "vpux/compiler/core/token_barrier_scheduler.hpp"
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

    // bool success = false;

    TokenBasedBarrierScheduler barrierScheduler(&ctx, func, _log, 4, 256);
    barrierScheduler.schedule();

    // Barrier Simulation
    // for (size_t barrier_bound = 4; !success && (barrier_bound >= 1UL); --barrier_bound) {
    //     TokenBasedBarrierScheduler barrierScheduler(&ctx, func, _log, barrier_bound, 256, numDmaEngines);
    //     barrierScheduler.schedule();

    //     // RuntimeSimulator simulator(&ctx, func, _log, numDmaEngines, 8);
    //     // success = simulator.assignPhysicalIDs();

    //     // std::cout << "Barrier simualtion result is " << success << " with upperbound " << barrier_bound <<
    //     std::endl;
    // }
}

}  // namespace

//
// createAssignVirtualBarriersPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createAssignVirtualBarriersPass(Logger log) {
    return std::make_unique<AssignVirtualBarriersPass>(log);
}
