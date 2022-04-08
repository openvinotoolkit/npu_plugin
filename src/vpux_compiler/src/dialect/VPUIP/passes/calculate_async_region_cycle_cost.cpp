//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/cost_model.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

namespace {

size_t calculateDMACycleCost(mlir::async::ExecuteOp asyncExec, VPU::ArchKind archKind,
                             const std::shared_ptr<VPUNN::VPUCostModel> costModel) {
    size_t cycleCost = 0;

    auto* bodyBlock = &asyncExec.body().front();
    for (auto& innerOp : bodyBlock->getOperations()) {
        if (auto nceClustOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(innerOp)) {
            cycleCost += calculateCopyCycles(nceClustOp.getInnerTaskOp(), archKind, costModel);
        } else {
            cycleCost += calculateCopyCycles(&innerOp, archKind, costModel);
        }
    }

    VPUX_THROW_UNLESS(cycleCost > 0, "Invalid cycle cost for 'async.execute' {0}", asyncExec);
    return cycleCost;
}

size_t getInnerOperationCycleCost(mlir::Operation* innerOp) {
    if (innerOp->hasAttr(DPUCost)) {
        return checked_cast<size_t>(innerOp->getAttr(DPUCost).cast<mlir::IntegerAttr>().getValue().getSExtValue());
    }
    return 0;
}

size_t retrieveDPUCycleCost(mlir::async::ExecuteOp asyncExec) {
    size_t cycleCost = 0;

    auto* bodyBlock = &asyncExec.body().front();
    for (auto& innerOp : bodyBlock->getOperations()) {
        if (auto nceClustOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(innerOp)) {
            cycleCost += getInnerOperationCycleCost(nceClustOp.getInnerTaskOp());
        } else {
            cycleCost += getInnerOperationCycleCost(&innerOp);
        }
    }

    VPUX_THROW_UNLESS(cycleCost > 0, "Invalid cycle cost for 'async.execute' {0}", asyncExec);
    return cycleCost;
}

size_t calculateUPACycles(mlir::Operation* innerOp) {
    // TODO calculate UPA cost
    VPUX_UNUSED(innerOp);
    if (!VPUIP::isPureViewOp(innerOp)) {
        // UPA cost not available set to 1
        return 1;
    }
    return 0;
}

size_t calculateUPACycleCost(mlir::async::ExecuteOp asyncExec) {
    size_t cycleCost = 0;

    auto* bodyBlock = &asyncExec.body().front();
    for (auto& innerOp : bodyBlock->getOperations()) {
        if (auto nceClustOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(innerOp)) {
            cycleCost += calculateUPACycles(nceClustOp.getInnerTaskOp());
        } else {
            cycleCost += calculateUPACycles(&innerOp);
        }
    }

    VPUX_THROW_UNLESS(cycleCost > 0, "Invalid cycle cost for 'async.execute' {0}", asyncExec);
    return cycleCost;
}

class CalculateAsyncRegionCycleCostPass final :
        public VPUIP::CalculateAsyncRegionCycleCostBase<CalculateAsyncRegionCycleCostPass> {
public:
    explicit CalculateAsyncRegionCycleCostPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void CalculateAsyncRegionCycleCostPass::safeRunOnFunc() {
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);
    const auto costModel = VPU::createCostModel(arch);

    func->walk([&](mlir::async::ExecuteOp asyncExec) {
        // calculate cycle cost based on executor
        if (!asyncExec->hasAttr(VPUIP::VPUIPDialect::getExecutorAttrName())) {
            return;
        }

        size_t cycleCost = 1;
        const auto executor = VPUIP::VPUIPDialect::getExecutorKind(asyncExec);

        if (executor == VPU::ExecutorKind::SHAVE_UPA) {
            // calculate UPA cycles
            cycleCost = calculateUPACycleCost(asyncExec);
        } else if (executor == VPU::ExecutorKind::SHAVE_ACT) {
            // calculate UPA cycles
            cycleCost = calculateUPACycleCost(asyncExec);
        } else if (executor == VPU::ExecutorKind::SHAVE_NN) {
            // calculate UPA cycles
            cycleCost = calculateUPACycleCost(asyncExec);
        } else if (executor == VPU::ExecutorKind::DMA_NN) {
            // calculate DMA cycles
            cycleCost = calculateDMACycleCost(asyncExec, arch, costModel);
        } else if (executor == VPU::ExecutorKind::NCE) {
            // DPU have cost calculated during workload generation, retrieve cost
            cycleCost = retrieveDPUCycleCost(asyncExec);
        } else if (executor == VPU::ExecutorKind::DPU) {
            // DPU have cost calculated during workload generation, retrieve cost
            cycleCost = retrieveDPUCycleCost(asyncExec);
        }
        // store cycle cost for async.execute
        asyncExec->setAttr(cycleCostAttrName, getIntAttr(asyncExec->getContext(), cycleCost));
    });
}

}  // namespace

//
// createCalculateAsyncRegionCycleCostPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createCalculateAsyncRegionCycleCostPass(Logger log) {
    return std::make_unique<CalculateAsyncRegionCycleCostPass>(log);
}
