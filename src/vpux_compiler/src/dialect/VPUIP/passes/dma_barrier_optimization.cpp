//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPURT/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

namespace {

class DMABarrierOptimizer final {
public:
    explicit DMABarrierOptimizer(mlir::FuncOp funcOp, Logger log): _func(funcOp), _log(log) {
    }
    void init();
    void optimize();

private:
    int64_t getDMAPortValue(VPURT::TaskOp taskOp);

    mlir::FuncOp _func;
    Logger _log;
    std::map<VPURT::DeclareVirtualBarrierOp, std::set<VPURT::TaskOp>> _barrierOpUpdateMap;
    std::map<VPURT::DeclareVirtualBarrierOp, std::set<VPURT::TaskOp>> _barrierOpWaitMap;
};

void DMABarrierOptimizer::init() {
    const auto updateBarrierConfigs = [&](VPURT::TaskOp taskOp) {
        for (const auto bar : taskOp.waitBarriers()) {
            auto barrierOp = bar.getDefiningOp<VPURT::DeclareVirtualBarrierOp>();
            VPUX_THROW_WHEN(barrierOp == nullptr, "Invalid barrier op type");
            _barrierOpUpdateMap[barrierOp].insert(taskOp);
        }
        for (const auto bar : taskOp.updateBarriers()) {
            auto barrierOp = bar.getDefiningOp<VPURT::DeclareVirtualBarrierOp>();
            VPUX_THROW_WHEN(barrierOp == nullptr, "Invalid barrier op type");
            _barrierOpWaitMap[barrierOp].insert(taskOp);
        }
    };
    _func->walk([&](VPURT::TaskOp taskOp) {
        updateBarrierConfigs(taskOp);
    });
}

/**
 * After DMA ops are unrolled, some divergent DMA ops will be assigned new port value while others remains default
 * port(0). So for the connected DMA ops using default DMA port, the related barrier can be removed then.
 */
void DMABarrierOptimizer::optimize() {
    _func->walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        auto barrierProducers = _barrierOpWaitMap[barrierOp];
        auto barrierConsumers = _barrierOpUpdateMap[barrierOp];
        bool barrierOnlyProducedByDMAPort0 = true;
        for (auto producer : barrierProducers) {
            if ((producer.getExecutorKind() != VPU::ExecutorKind::DMA_NN) || (getDMAPortValue(producer) != 0)) {
                barrierOnlyProducedByDMAPort0 = false;
                break;
            }
        }

        bool barrierOnlyConsumedByDMAPort0 = true;
        for (auto consumer : barrierConsumers) {
            if ((consumer.getExecutorKind() != VPU::ExecutorKind::DMA_NN) || (getDMAPortValue(consumer) != 0)) {
                barrierOnlyConsumedByDMAPort0 = false;
                break;
            }
        }

        if (barrierOnlyProducedByDMAPort0 && barrierOnlyConsumedByDMAPort0) {
            for (auto producer : barrierProducers) {
                auto iter = llvm::find_if(producer.updateBarriers(), [&barrierOp](mlir::Value bar) {
                    return barrierOp = bar.getDefiningOp<VPURT::DeclareVirtualBarrierOp>();
                });
                auto index = static_cast<unsigned>(std::distance(producer.updateBarriers().begin(), iter));
                producer.updateBarriersMutable().erase(index);
            }

            for (auto consumer : barrierConsumers) {
                auto iter = llvm::find_if(consumer.waitBarriers(), [&barrierOp](mlir::Value bar) {
                    return bar.getDefiningOp<VPURT::DeclareVirtualBarrierOp>() == barrierOp;
                });
                auto index = static_cast<unsigned>(std::distance(consumer.waitBarriers().begin(), iter));
                consumer.waitBarriersMutable().erase(index);
            }

            _barrierOpUpdateMap.erase(barrierOp);
            _barrierOpWaitMap.erase(barrierOp);
            barrierOp->dropAllUses();
            barrierOp.erase();
        }
    });
}

int64_t DMABarrierOptimizer::getDMAPortValue(VPURT::TaskOp taskOp) {
    auto* wrappedTaskOp = taskOp.getInnerTaskOp();
    if (auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(wrappedTaskOp)) {
        wrappedTaskOp = clusterTilingOp.getInnerTaskOp();
    }
    if (auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(wrappedTaskOp)) {
        return dmaOp.port();
    } else if (auto compressedDmaOp = mlir::dyn_cast<VPUIP::CompressedDMAOp>(wrappedTaskOp)) {
        return compressedDmaOp.port();
    } else if (auto depthToSpaceDMAOp = mlir::dyn_cast<VPUIP::DepthToSpaceDMAOp>(wrappedTaskOp)) {
        return depthToSpaceDMAOp.port();
    } else if (auto permuteDMAOp = mlir::dyn_cast<VPUIP::PermuteDMAOp>(wrappedTaskOp)) {
        return permuteDMAOp.port();
    } else {
        VPUX_THROW("Could not cast to DMA task");
    }
}

//
//  DMABarrierOptimizationPass
//

/**
 * For device using multi dma ports, the dma related barrier optimization is disabled since the barrier scheduler
 * doesn't known the dma op's port value. In this pass, the port value is already assigned so further optimization can
 * be done.
 * 1. barrier which is produced and consumed by DMA with port 0 will be removed.
 *    DMA0 port0 -> barrier -> DMA1 port0
 *
 * 2. unrolled DMA ops with different ports will keep unchanged.
 *    DMA0 Cluster 0 port0 \       / DMA1 Cluster 0 port0
 *                          barrier
 *    DMA0 Cluster 1 port1 /       \ DMA1 Cluster 1 port1
 *
 */
class DMABarrierOptimizationPass final : public VPUIP::DMABarrierOptimizationBase<DMABarrierOptimizationPass> {
public:
    explicit DMABarrierOptimizationPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void DMABarrierOptimizationPass::safeRunOnFunc() {
    auto func = getFunction();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    auto dmaOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.count();

    if (dmaPortCount < 2) {
        _log.trace("DMABarrierOptimization is enabled only when have multi dma port. Got: {0}", dmaPortCount);
        return;
    }
    DMABarrierOptimizer optimizer(func, _log);
    optimizer.init();
    optimizer.optimize();
}

}  // namespace

//
// createUnrollPermuteToNNDMAPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createDMABarrierOptimizationPass(Logger log) {
    return std::make_unique<DMABarrierOptimizationPass>(log);
}
