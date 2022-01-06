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

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
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

namespace {

const EnumMap<VPU::ArchKind, int64_t> MAX_BARRIERS_PER_INFERENCE = {
        {VPU::ArchKind::KMB, 64 / 2},  // half barries are used (runtime limitation)
        {VPU::ArchKind::TBH, 64 / 2},  // half barries are used (runtime limitation)
        {VPU::ArchKind::MTL, 64},      //
        {VPU::ArchKind::LNL, 64},      //
};

int64_t getNumAvailableBarriers(mlir::Operation* parentOp) {
    const auto arch = VPU::getArch(parentOp);

    auto module = parentOp->getParentOfType<mlir::ModuleOp>();
    auto nceResOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::NCE);
    VPUX_THROW_UNLESS(nceResOp != nullptr, "Failed to get NCE Executor information");

    const auto numClusters = nceResOp.count();

    const auto maxNumClustersForArch = VPU::getMaxDPUClusterNum(module);
    VPUX_THROW_UNLESS(maxNumClustersForArch != 0, "Failed to get maxNumClustersForArch");

    const auto barIt = MAX_BARRIERS_PER_INFERENCE.find(arch);
    VPUX_THROW_WHEN(barIt == MAX_BARRIERS_PER_INFERENCE.end(), "Unsupported VPU architecture '{0}'", arch);

    const auto maxBarriersPerInference = barIt->second;

    const auto barriersPerCluster = maxBarriersPerInference / maxNumClustersForArch;
    const auto maxNumBarriers = std::min(maxBarriersPerInference, barriersPerCluster * numClusters);

    return maxNumBarriers;
}

}  // namespace

void AssignVirtualBarriersPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    auto numBarriersToUse = numBarriers.hasValue() ? numBarriers.getValue() : getNumAvailableBarriers(func);
    static constexpr int64_t MAX_SLOT_COUNT = 256;
    auto numSlotsPerBarrierToUse = numSlotsPerBarrier.hasValue() ? numSlotsPerBarrier.getValue() : MAX_SLOT_COUNT;

    VPURT::FeasibleBarrierScheduler barrierScheduler(&ctx, func, _log);
    barrierScheduler.init();

    bool success = false;
    for (size_t barrier_bound = (numBarriersToUse / 2); !success && (barrier_bound >= 1UL); --barrier_bound) {
        barrierScheduler.schedule(barrier_bound, numSlotsPerBarrierToUse);
        success = barrierScheduler.performRuntimeSimulation();
    }
    barrierScheduler.clearUniqueID();

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
