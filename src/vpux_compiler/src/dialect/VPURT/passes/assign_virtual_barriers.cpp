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

#include "vpux/compiler/core/runtime_simulator.hpp"
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
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto resOp = IERT::RunTimeResourcesOp::getFromModule(module);

    const auto dmaAttr = VPU::ExecutorKindAttr::get(&ctx, VPU::ExecutorKind::DMA_NN);
    auto dmaResOp = resOp.getExecutor(dmaAttr);
    VPUX_THROW_UNLESS(dmaResOp != nullptr, "Failed to get DMA_NN information");

    const auto numDmaEngines = dmaResOp.count();
    VPUX_THROW_UNLESS(numDmaEngines <= MAX_DMA_ENGINES, "Found {0} DMA engines (max {1})", numDmaEngines,
                      MAX_DMA_ENGINES);

    // bool success = false;

    // Barrier Simulation
    // for (size_t barrier_bound = 4; !success && (barrier_bound >= 1UL); --barrier_bound) {
    TokenBasedBarrierScheduler barrierScheduler(&ctx, func, 4, 256);
    barrierScheduler.schedule();

    // RuntimeSimulator simulator(&ctx, func, _log, numDmaEngines, 8);
    // success = simulator.assignPhysicalIDs();

    // std::cout << "Barrier simualtion result is " << success << " with upperbound " << barrier_bound << std::endl;
    // }
}

}  // namespace

//
// createAssignVirtualBarriersPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createAssignVirtualBarriersPass(Logger log) {
    return std::make_unique<AssignVirtualBarriersPass>(log);
}
