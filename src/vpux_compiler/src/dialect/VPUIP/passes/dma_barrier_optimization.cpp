//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/dialect/VPURT/attributes.hpp"
#include "vpux/compiler/dialect/VPURT/barrier_scheduler.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/dma.hpp"

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
    std::set<int64_t> getPorts(std::set<vpux::VPURT::TaskOp> tasks);
    bool useSamePort(std::set<vpux::VPURT::TaskOp> producers, vpux::VPURT::TaskOp consumer);
    size_t countProducerTasksToBarrier(VPURT::TaskOp op);
    bool canBarriersBeMerged(VPURT::DeclareVirtualBarrierOp barrier1, VPURT::DeclareVirtualBarrierOp barrier2);
    void insertBarrierToUpdateList(VPURT::TaskOp& op, VPURT::DeclareVirtualBarrierOp& barrierOp);
    void insertBarrierToWaitList(VPURT::TaskOp& op, VPURT::DeclareVirtualBarrierOp& barrierOp);
    bool doesControlPathExist(vpux::VPURT::TaskOp task1, vpux::VPURT::TaskOp task2);
    void removeRedundantDependenciesForDMA();
    void mergeBarriers();
    size_t getSmallestCycleStart(VPURT::DeclareVirtualBarrierOp& barrierOp);

    struct sortBarriersByIROrder {
        bool operator()(const VPURT::DeclareVirtualBarrierOp& a, const VPURT::DeclareVirtualBarrierOp& b) const {
            return a->isBeforeInBlock(b);
        }
    };

    struct sortTasksByIROrder {
        bool operator()(const VPURT::TaskOp& a, const VPURT::TaskOp& b) const {
            return a->isBeforeInBlock(b);
        }
    };

    mlir::FuncOp _func;
    Logger _log;
    std::map<VPURT::DeclareVirtualBarrierOp, std::set<VPURT::TaskOp>, sortBarriersByIROrder> _barrierOpUpdateMap;
    std::map<VPURT::DeclareVirtualBarrierOp, std::set<VPURT::TaskOp>, sortBarriersByIROrder> _barrierOpWaitMap;
    std::map<VPURT::TaskOp, std::set<VPURT::DeclareVirtualBarrierOp>, sortTasksByIROrder> _taskOpWaitMap{};
    std::map<VPURT::TaskOp, std::set<VPURT::DeclareVirtualBarrierOp>, sortTasksByIROrder> _taskOpUpdateMap{};
    // The indexes of first vector are dma ports
    SmallVector<SmallVector<VPURT::TaskOp>> orderedDMATasks{};
};

void DMABarrierOptimizer::init() {
    auto module = _func->getParentOfType<mlir::ModuleOp>();
    auto dmaOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOp.count();
    orderedDMATasks.resize(dmaPortCount);

    const auto updateBarrierConfigs = [&](VPURT::TaskOp taskOp) {
        for (const auto bar : taskOp.waitBarriers()) {
            auto barrierOp = bar.getDefiningOp<VPURT::DeclareVirtualBarrierOp>();
            VPUX_THROW_WHEN(barrierOp == nullptr, "Invalid barrier op type");
            _barrierOpUpdateMap[barrierOp].insert(taskOp);
            _taskOpWaitMap[taskOp].insert(barrierOp);
        }
        for (const auto bar : taskOp.updateBarriers()) {
            auto barrierOp = bar.getDefiningOp<VPURT::DeclareVirtualBarrierOp>();
            VPUX_THROW_WHEN(barrierOp == nullptr, "Invalid barrier op type");
            _barrierOpWaitMap[barrierOp].insert(taskOp);
            _taskOpUpdateMap[taskOp].insert(barrierOp);
        }
    };
    _func->walk([&](VPURT::TaskOp taskOp) {
        updateBarrierConfigs(taskOp);

        if (taskOp.getExecutorKind() == VPU::ExecutorKind::DMA_NN) {
            orderedDMATasks[getDMAPortValue(taskOp)].push_back(taskOp);
        }
    });
}

std::set<int64_t> DMABarrierOptimizer::getPorts(std::set<vpux::VPURT::TaskOp> tasks) {
    std::set<int64_t> ports;
    for (auto task : tasks) {
        ports.insert(getDMAPortValue(task));
    }

    return ports;
}

/// @brief all DMAs are using same port
bool DMABarrierOptimizer::useSamePort(std::set<vpux::VPURT::TaskOp> producers, vpux::VPURT::TaskOp consumer) {
    std::set<int64_t> producerPorts = getPorts(producers);
    producerPorts.erase(getDMAPortValue(consumer));

    return producerPorts.empty();
}

// This function returns the number of producers to a barrier.
// On VPU H/W, a NCE task is executed across multiple DPUs via workloads descriptors (known as variants).
// Each variant must update the barrier to signal that is is complete.
// An NCE task may have up 50 workloads descriptors (which are generated in the NCE DPU workloads pass).
// Therefore, the number of variants must be retrieved here as they will all update a barrier and
// contribute to the 256 producer limit that a barrier has.
// A DMA/UPA does not have variants, therefore they always just requires 1 producer slot to a barrier.
size_t DMABarrierOptimizer::countProducerTasksToBarrier(VPURT::TaskOp op) {
    if (op.getExecutorKind() == VPU::ExecutorKind::NCE) {
        auto innerTaskOp = op.getInnerTaskOp();
        if (auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(innerTaskOp)) {
            innerTaskOp = clusterTilingOp.getInnerTaskOp();
        }
        auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(innerTaskOp);
        VPUX_THROW_UNLESS(nceOp != nullptr, "Could not cast to NCE task");
        return nceOp.getNumVariants();
    }

    if (op.getExecutorKind() == VPU::ExecutorKind::DMA_NN || op.getExecutorKind() == VPU::ExecutorKind::SHAVE_UPA ||
        op.getExecutorKind() == VPU::ExecutorKind::SHAVE_ACT) {
        return 1;
    }

    VPUX_THROW("This operation does not run on hardware");
}

/// @brief check if two barriers can be merged
/// @details two barriers A and B can be merged if
/// 1. any producer of barrier A controls any consumer of barrier B
/// 2. any producer of barrier B controls any consumer of barrier A
bool DMABarrierOptimizer::canBarriersBeMerged(VPURT::DeclareVirtualBarrierOp barrier1,
                                              VPURT::DeclareVirtualBarrierOp barrier2) {
    auto barrierProducers1 = _barrierOpWaitMap[barrier1];
    auto barrierConsumers1 = _barrierOpUpdateMap[barrier1];
    auto barrierProducers2 = _barrierOpWaitMap[barrier2];
    auto barrierConsumers2 = _barrierOpUpdateMap[barrier2];

    if (barrierProducers1.empty() || barrierConsumers1.empty() || barrierProducers2.empty() ||
        barrierConsumers2.empty()) {
        return false;
    }

    for (auto producer : barrierProducers1) {
        for (auto consumer : barrierConsumers2) {
            if ((!barrierConsumers1.count(consumer)) &&
                ((producer.getExecutorKind() != VPU::ExecutorKind::DMA_NN) ||
                 (consumer.getExecutorKind() != VPU::ExecutorKind::DMA_NN) || (!producer->isBeforeInBlock(consumer)))) {
                return false;
            }
        }
    }

    for (auto producer : barrierProducers2) {
        for (auto consumer : barrierConsumers1) {
            if ((!barrierConsumers2.count(consumer)) &&
                ((producer.getExecutorKind() != VPU::ExecutorKind::DMA_NN) ||
                 (consumer.getExecutorKind() != VPU::ExecutorKind::DMA_NN) || (!producer->isBeforeInBlock(consumer)))) {
                return false;
            }
        }
    }

    return true;
}

void DMABarrierOptimizer::insertBarrierToUpdateList(VPURT::TaskOp& op, VPURT::DeclareVirtualBarrierOp& barrierOp) {
    SmallVector<VPURT::DeclareVirtualBarrierOp> orderedBarrierList;
    for (auto bar : op.updateBarriers()) {
        orderedBarrierList.push_back(bar.getDefiningOp<VPURT::DeclareVirtualBarrierOp>());
    }

    orderedBarrierList.push_back(barrierOp);
    std::sort(orderedBarrierList.begin(), orderedBarrierList.end(), [](const auto& lhs, const auto& rhs) {
        return lhs->isBeforeInBlock(rhs);
    });

    op.updateBarriersMutable().clear();
    for (auto bar : orderedBarrierList) {
        op.updateBarriersMutable().append(bar.barrier());
    }
}

void DMABarrierOptimizer::insertBarrierToWaitList(VPURT::TaskOp& op, VPURT::DeclareVirtualBarrierOp& barrierOp) {
    SmallVector<VPURT::DeclareVirtualBarrierOp> orderedBarrierList;
    for (auto bar : op.waitBarriers()) {
        orderedBarrierList.push_back(bar.getDefiningOp<VPURT::DeclareVirtualBarrierOp>());
    }

    orderedBarrierList.push_back(barrierOp);
    std::sort(orderedBarrierList.begin(), orderedBarrierList.end(), [](const auto& lhs, const auto& rhs) {
        return lhs->isBeforeInBlock(rhs);
    });

    op.waitBarriersMutable().clear();
    for (auto bar : orderedBarrierList) {
        op.waitBarriersMutable().append(bar.barrier());
    }
}

/// @brief check if task2 executes after task1
bool DMABarrierOptimizer::doesControlPathExist(vpux::VPURT::TaskOp task1, vpux::VPURT::TaskOp task2) {
    if ((task1.getExecutorKind() != VPU::ExecutorKind::DMA_NN) ||
        (task2.getExecutorKind() != VPU::ExecutorKind::DMA_NN)) {
        return false;
    }

    if (getDMAPortValue(task1) != getDMAPortValue(task2)) {
        return false;
    }

    if (task2->isBeforeInBlock(task1)) {
        return false;
    }

    return true;
}

/// @brief remove redundant dependency from a barrier
/// @details For two producers {a, b} of a barrier, if a depends on b then b isn't a necessary producer for this barrier
/// For two consumers {a, b} of a barrier, if a depends on b then a isn't a necessary consumer for this barrier
void DMABarrierOptimizer::removeRedundantDependenciesForDMA() {
    _func->walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        auto& barrierProducers = _barrierOpWaitMap[barrierOp];
        auto& barrierConsumers = _barrierOpUpdateMap[barrierOp];

        std::set<vpux::VPURT::TaskOp> producersToRemove;
        auto _barrierProducerItr = barrierProducers.begin();
        for (; _barrierProducerItr != barrierProducers.end(); _barrierProducerItr++) {
            auto _nextbarrierProducerItr = _barrierProducerItr;
            _nextbarrierProducerItr++;
            for (; _nextbarrierProducerItr != barrierProducers.end(); _nextbarrierProducerItr++) {
                if (doesControlPathExist(*_barrierProducerItr, *_nextbarrierProducerItr)) {
                    producersToRemove.insert(*_barrierProducerItr);
                } else if (doesControlPathExist(*_nextbarrierProducerItr, *_barrierProducerItr)) {
                    producersToRemove.insert(*_nextbarrierProducerItr);
                }
            }
        }

        for (auto producer : producersToRemove) {
            auto iter = llvm::find_if(producer.updateBarriers(), [&barrierOp](mlir::Value bar) {
                return barrierOp == bar.getDefiningOp<VPURT::DeclareVirtualBarrierOp>();
            });
            auto index = static_cast<unsigned>(std::distance(producer.updateBarriers().begin(), iter));
            producer.updateBarriersMutable().erase(index);
            barrierProducers.erase(producer);
            _taskOpUpdateMap[producer].erase(barrierOp);

            _log.trace("Remove producer task {0} for barrier {1}", producer, barrierOp);
        }

        std::set<vpux::VPURT::TaskOp> consumersToRemove;
        auto _barrierConsumerItr = barrierConsumers.begin();
        for (; _barrierConsumerItr != barrierConsumers.end(); _barrierConsumerItr++) {
            auto _nextbarrierConsumerItr = _barrierConsumerItr;
            _nextbarrierConsumerItr++;
            for (; _nextbarrierConsumerItr != barrierConsumers.end(); _nextbarrierConsumerItr++) {
                if (doesControlPathExist(*_barrierConsumerItr, *_nextbarrierConsumerItr)) {
                    consumersToRemove.insert(*_nextbarrierConsumerItr);
                } else if (doesControlPathExist(*_nextbarrierConsumerItr, *_barrierConsumerItr)) {
                    consumersToRemove.insert(*_barrierConsumerItr);
                }
            }
        }

        for (auto consumer : consumersToRemove) {
            auto iter = llvm::find_if(consumer.waitBarriers(), [&barrierOp](mlir::Value bar) {
                return bar.getDefiningOp<VPURT::DeclareVirtualBarrierOp>() == barrierOp;
            });
            auto index = static_cast<unsigned>(std::distance(consumer.waitBarriers().begin(), iter));
            consumer.waitBarriersMutable().erase(index);
            barrierConsumers.erase(consumer);
            _taskOpWaitMap[consumer].erase(barrierOp);

            _log.trace("Remove consumer task {0} for barrier {1}", consumer, barrierOp);
        }
    });
}

size_t DMABarrierOptimizer::getSmallestCycleStart(VPURT::DeclareVirtualBarrierOp& barrierOp) {
    size_t smallestCycleStart = std::numeric_limits<size_t>::max();
    for (auto producer : _barrierOpWaitMap[barrierOp]) {
        if (producer->hasAttr(cycleBegin)) {
            auto currCycle = producer->getAttr(cycleBegin).cast<mlir::IntegerAttr>().getValue().getSExtValue();
            smallestCycleStart = std::min(smallestCycleStart, checked_cast<size_t>(currCycle));
        } else {
            VPUX_THROW("TaskOp {0} was not assigned a start cycle time by the CMX memory scheduler", producer);
        }
    }

    return smallestCycleStart;
}

void DMABarrierOptimizer::mergeBarriers() {
    // Remove explicit barrier dependency between DMAs before merging barriers
    // 1) if a barrier only has DMAs using single port as its producer, remove all DMAs using the same port from its
    // consumers
    // 2) if a barrier only has DMAs using single port as its consumer, remove all DMAs using the same port from its
    // producers
    _func->walk([&](VPURT::DeclareVirtualBarrierOp barrierOp) {
        auto& barrierProducers = _barrierOpWaitMap[barrierOp];
        auto& barrierConsumers = _barrierOpUpdateMap[barrierOp];

        bool barrierOnlyProducedByDMA = true;
        for (auto producer : barrierProducers) {
            if ((producer.getExecutorKind() != VPU::ExecutorKind::DMA_NN)) {
                barrierOnlyProducedByDMA = false;
                break;
            }
        }

        if (barrierOnlyProducedByDMA) {
            std::set<vpux::VPURT::TaskOp> consumersToRemove;
            for (auto cons : barrierConsumers) {
                if ((cons.getExecutorKind() == VPU::ExecutorKind::DMA_NN) && useSamePort(barrierProducers, cons)) {
                    consumersToRemove.insert(cons);
                }
            }

            for (auto consumer : consumersToRemove) {
                auto iter = llvm::find_if(consumer.waitBarriers(), [&barrierOp](mlir::Value bar) {
                    return bar.getDefiningOp<VPURT::DeclareVirtualBarrierOp>() == barrierOp;
                });
                auto index = static_cast<unsigned>(std::distance(consumer.waitBarriers().begin(), iter));
                consumer.waitBarriersMutable().erase(index);
                barrierConsumers.erase(consumer);
                _taskOpWaitMap[consumer].erase(barrierOp);

                _log.trace("Remove consumer task {0} for barrier {1}", consumer, barrierOp);
            }
        }

        bool barrierOnlyConsumedByDMA = true;
        for (auto consumer : barrierConsumers) {
            if ((consumer.getExecutorKind() != VPU::ExecutorKind::DMA_NN)) {
                barrierOnlyConsumedByDMA = false;
                break;
            }
        }

        if (barrierOnlyConsumedByDMA) {
            std::set<vpux::VPURT::TaskOp> producersToRemove;

            if (barrierConsumers.empty()) {
                producersToRemove = barrierProducers;
            } else {
                for (auto prod : barrierProducers) {
                    if ((prod.getExecutorKind() == VPU::ExecutorKind::DMA_NN) && useSamePort(barrierConsumers, prod)) {
                        producersToRemove.insert(prod);
                    }
                }
            }

            for (auto producer : producersToRemove) {
                auto iter = llvm::find_if(producer.updateBarriers(), [&barrierOp](mlir::Value bar) {
                    return barrierOp == bar.getDefiningOp<VPURT::DeclareVirtualBarrierOp>();
                });
                auto index = static_cast<unsigned>(std::distance(producer.updateBarriers().begin(), iter));
                producer.updateBarriersMutable().erase(index);
                barrierProducers.erase(producer);
                _taskOpUpdateMap[producer].erase(barrierOp);

                _log.trace("Remove producer task {0} for barrier {1}", producer, barrierOp);
            }
        }

        if (barrierProducers.empty() && barrierConsumers.empty()) {
            _barrierOpUpdateMap.erase(barrierOp);
            _barrierOpWaitMap.erase(barrierOp);
            barrierOp->dropAllUses();
            barrierOp.erase();
        } else if (barrierProducers.empty() || barrierConsumers.empty()) {
            VPUX_THROW("Invalid optimization : Only producer or comsumer list become empty for barrier {0}", barrierOp);
        }
    });

    // Merge barriers if possible
    std::set<VPURT::DeclareVirtualBarrierOp> barriersToRemove;
    auto _barrierOpUpdateMapItr = _barrierOpUpdateMap.begin();
    for (; _barrierOpUpdateMapItr != _barrierOpUpdateMap.end(); _barrierOpUpdateMapItr++) {
        auto& currentConsumers = _barrierOpUpdateMapItr->second;
        if (!(currentConsumers.empty())) {
            auto _nextBarrierOpUpdateMapItr = _barrierOpUpdateMapItr;
            _nextBarrierOpUpdateMapItr++;

            for (; _nextBarrierOpUpdateMapItr != _barrierOpUpdateMap.end(); _nextBarrierOpUpdateMapItr++) {
                auto currentBarrier = _barrierOpUpdateMapItr->first;
                auto nextBarrier = _nextBarrierOpUpdateMapItr->first;
                if (canBarriersBeMerged(currentBarrier, nextBarrier)) {
                    auto& currentProducers = _barrierOpWaitMap[currentBarrier];
                    auto& nextConsumers = _nextBarrierOpUpdateMapItr->second;
                    auto& nextProducers = _barrierOpWaitMap[nextBarrier];
                    size_t variantsCount = 0;
                    size_t invariantsCount = 0;
                    for (auto oldProducer : currentProducers) {
                        variantsCount += countProducerTasksToBarrier(oldProducer);
                        invariantsCount++;
                    }
                    for (auto newProducer : nextProducers) {
                        variantsCount += countProducerTasksToBarrier(newProducer);
                        invariantsCount++;
                    }

                    auto module = _func->getParentOfType<mlir::ModuleOp>();
                    auto nceOp = IE::getAvailableExecutor(module, VPU::ExecutorKind::NCE);
                    auto numClusters = nceOp.count();
                    // ClusterTiling is Unrolled. So set invariants limits as 64 for multiple clusters.
                    size_t invariantsLimits = (numClusters == 1) ? 32 : 64;
                    if (((variantsCount <= 255) && (invariantsCount <= invariantsLimits))) {
                        for (auto producer : nextProducers) {
                            auto iter = llvm::find_if(producer.updateBarriers(), [&nextBarrier](mlir::Value bar) {
                                return bar.getDefiningOp<VPURT::DeclareVirtualBarrierOp>() == nextBarrier;
                            });
                            auto index = static_cast<unsigned>(std::distance(producer.updateBarriers().begin(), iter));
                            producer.updateBarriersMutable().erase(index);

                            if (!currentProducers.count(producer)) {
                                insertBarrierToUpdateList(producer, currentBarrier);
                            }

                            currentProducers.insert(producer);
                            _taskOpUpdateMap[producer].erase(nextBarrier);
                            _taskOpUpdateMap[producer].insert(currentBarrier);
                        }

                        for (auto consumer : nextConsumers) {
                            auto iter = llvm::find_if(consumer.waitBarriers(), [&nextBarrier](mlir::Value bar) {
                                return bar.getDefiningOp<VPURT::DeclareVirtualBarrierOp>() == nextBarrier;
                            });
                            auto index = static_cast<unsigned>(std::distance(consumer.waitBarriers().begin(), iter));
                            consumer.waitBarriersMutable().erase(index);

                            if (!currentConsumers.count(consumer)) {
                                insertBarrierToWaitList(consumer, currentBarrier);
                            }

                            currentConsumers.insert(consumer);
                            _taskOpWaitMap[consumer].erase(nextBarrier);
                            _taskOpWaitMap[consumer].insert(currentBarrier);
                        }

                        nextProducers.clear();
                        nextConsumers.clear();
                        barriersToRemove.insert(nextBarrier);
                    }
                }
            }
        }
    }

    // Remove unused barriers
    for (auto barrierOp : barriersToRemove) {
        _barrierOpUpdateMap.erase(barrierOp);
        _barrierOpWaitMap.erase(barrierOp);
        barrierOp->dropAllUses();
        barrierOp.erase();
    }
}

/// @brief remove redundant dependency to barrier and merge barriers
/// @details after DMA ops are unrolled, some divergent DMA ops will be assigned new port value while others remains
/// default port(0). So we can further optimize barriers related to DMA.
/// 1. if a barrier only has DMA as producer task, we can remove the barrier's DMA consumer if the producer DMA doesn't
/// need a barrier to control the consumer DMA, e.g. they are using same DMA port or they are accessing different memory
/// chunck.
/// 2. merge barriers as much as possible after step 1
/// 3. do step 1 again because there maybe redundant dependency after we merge barriers
void DMABarrierOptimizer::optimize() {
    removeRedundantDependenciesForDMA();

    mergeBarriers();

    // There maybe redundant dependency again after we merge barriers.
    removeRedundantDependenciesForDMA();

    SmallVector<VPURT::DeclareVirtualBarrierOp> orderedBarriersByCycle;
    for (auto& barrierOp : _barrierOpWaitMap) {
        orderedBarriersByCycle.push_back(barrierOp.first);
    }

    std::sort(orderedBarriersByCycle.begin(), orderedBarriersByCycle.end(), [&](auto& lhs, auto& rhs) {
        return getSmallestCycleStart(lhs) < getSmallestCycleStart(rhs);
    });

    // reorder barrier by cycle
    VPURT::DeclareVirtualBarrierOp preBarrier = nullptr;
    for (auto& curBarrier : orderedBarriersByCycle) {
        if (preBarrier) {
            curBarrier->moveAfter(preBarrier);
        }
        preBarrier = curBarrier;
    }
}

int64_t DMABarrierOptimizer::getDMAPortValue(VPURT::TaskOp taskOp) {
    auto* wrappedTaskOp = taskOp.getInnerTaskOp();
    if (auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(wrappedTaskOp)) {
        wrappedTaskOp = clusterTilingOp.getInnerTaskOp();
    }
    return vpux::getDMAPortValue(wrappedTaskOp);
}

//
//  DMABarrierOptimizationPass
//

/**
 * For device using multi dma ports, the dma related barrier optimization is disabled since the barrier scheduler
 * doesn't known the dma op's port value. In this pass, the port value is already assigned so further optimization can
 * be done.
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
