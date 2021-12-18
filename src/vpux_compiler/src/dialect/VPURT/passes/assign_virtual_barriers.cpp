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

#include "vpux/compiler/core/feasible_schedule_generator.hpp"
#include "vpux/compiler/core/runtime_simulator.hpp"
#include "vpux/compiler/dialect/VPURT/passes.hpp"

#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/DenseMap.h>

using namespace vpux;

namespace {

// Same value for all architectures for now
constexpr int64_t MAX_BARRIERS_FOR_ARCH = 64;
constexpr int64_t MAX_PRODUCERS_PER_BARRIER = 256;

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

    const auto nceAttr = VPU::ExecutorKindAttr::get(&ctx, VPU::ExecutorKind::NCE);
    auto nceResOp = resOp.getExecutor(nceAttr);
    VPUX_THROW_UNLESS(nceResOp != nullptr, "Failed to get NCE Executor information");

    const auto numClusters = nceResOp.count();

    const auto maxNumClustersForArch = VPU::getMaxDPUClusterNum(module);
    VPUX_THROW_UNLESS(maxNumClustersForArch != 0, "Failed to get maxNumClustersForArch");

    constexpr auto maxBarriersPerInference = MAX_BARRIERS_FOR_ARCH / 2;  // half barries are used
    const auto barriersPerCluster = maxBarriersPerInference / maxNumClustersForArch;
    const auto maxNumBarriers = std::min(maxBarriersPerInference, barriersPerCluster * numClusters);

    const auto numBarriers = _numBarriersOpt.hasValue() ? _numBarriersOpt.getValue() : maxNumBarriers / 2;
    VPUX_THROW_UNLESS(numBarriers > 0 && numBarriers <= maxNumBarriers,
                      "Number of physical barriers '{0}' is out of range '{1}'", numBarriers, maxNumBarriers);

    // Barrier scheduler
    // TokenBasedBarrierScheduler barrierScheduler(&ctx, func, numBarriers, MAX_PRODUCERS_PER_BARRIER, _log);
    // barrierScheduler.schedule();

    FeasibleBarrierScheduler(&ctx, func, _log, numBarriers, MAX_PRODUCERS_PER_BARRIER);
}

}  // namespace

//
// createAssignVirtualBarriersPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createAssignVirtualBarriersPass(Logger log) {
    return std::make_unique<AssignVirtualBarriersPass>(log);
}
