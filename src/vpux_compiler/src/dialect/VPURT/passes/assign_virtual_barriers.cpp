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
//#include "vpux/compiler/dialect/VPUIP/attributes/arch.hpp"
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

    //Barrier scheduler
    TokenBasedBarrierScheduler barrierScheduler(&ctx,func, 4, 256);
    barrierScheduler.schedule();

}

}  // namespace

//
// createAssignVirtualBarriersPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createAssignVirtualBarriersPass(Logger log) {
    return std::make_unique<AssignVirtualBarriersPass>(log);
}
