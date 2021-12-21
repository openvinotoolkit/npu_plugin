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

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/range.hpp"

using namespace vpux;
static constexpr StringLiteral schedulingNumberAttrName = "SchedulingNumber";
//
// Constructor
//

TokenBasedBarrierScheduler::TokenBasedBarrierScheduler(mlir::MLIRContext* ctx, mlir::FuncOp func, Logger log,
                                                       int64_t numBarriers, int64_t slotCount, int64_t numDmaEngines)
        : _ctx(ctx),
          _func(func),
          _log(log),
          builder(_func.getBody()),
          barrierCount_(numBarriers),
          slotCount_(slotCount),
          _numDmaEngines(numDmaEngines) {
    saveOriginalDependency();
}

void TokenBasedBarrierScheduler::getTaskOpUpdateWaitMap(
        std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>>&
                barrierOpUpdateWaitMap,
        std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>>&
                taskOpUpdateWaitMap) {
    for (auto iter = barrierOpUpdateWaitMap.begin(); iter != barrierOpUpdateWaitMap.end(); iter++) {
        auto barrierOp = (*iter).first;
        auto producers = (*iter).second.first;
        auto consumers = (*iter).second.second;
        for (auto prod = producers.begin(); prod != producers.end(); prod++) {
            auto taskUpateItr = taskOpUpdateWaitMap.find(*prod);
            if (taskUpateItr != taskOpUpdateWaitMap.end()) {
                taskUpateItr->second.second.insert(barrierOp);
            } else {
                std::set<mlir::Operation*> newBarrierProducers{};
                std::set<mlir::Operation*> newBarrierConsumers{barrierOp};
                taskOpUpdateWaitMap.insert(
                        std::make_pair(*prod, std::make_pair(newBarrierProducers, newBarrierConsumers)));
            }
        }

        for (auto cons = consumers.begin(); cons != consumers.end(); cons++) {
            auto taskWaitItr = taskOpUpdateWaitMap.find(*cons);
            if (taskWaitItr != taskOpUpdateWaitMap.end()) {
                taskWaitItr->second.first.insert(barrierOp);
            } else {
                std::set<mlir::Operation*> newBarrierProducers{barrierOp};
                std::set<mlir::Operation*> newBarrierConsumers{};
                taskOpUpdateWaitMap.insert(
                        std::make_pair(*cons, std::make_pair(newBarrierProducers, newBarrierConsumers)));
            }
        }
    }
}

void TokenBasedBarrierScheduler::getTaskOpUpdateWaitMap(
        std::map<mlir::Operation*,
                 std::pair<std::set<mlir::Operation*, task_operation_comparator_t>,
                           std::set<mlir::Operation*, task_operation_comparator_t>>,
                 operation_comparator_t>& barrierOpUpdateWaitMap,
        std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>,
                 task_operation_comparator_by_schedule_time_t>& taskOpUpdateWaitMap) {
    for (auto iter = barrierOpUpdateWaitMap.begin(); iter != barrierOpUpdateWaitMap.end(); iter++) {
        auto barrierOp = (*iter).first;
        auto producers = (*iter).second.first;
        auto consumers = (*iter).second.second;
        for (auto prod = producers.begin(); prod != producers.end(); prod++) {
            auto taskUpateItr = taskOpUpdateWaitMap.find(*prod);
            if (taskUpateItr != taskOpUpdateWaitMap.end()) {
                taskUpateItr->second.second.insert(barrierOp);
            } else {
                std::set<mlir::Operation*> newBarrierProducers{};
                std::set<mlir::Operation*> newBarrierConsumers{barrierOp};
                taskOpUpdateWaitMap.insert(
                        std::make_pair(*prod, std::make_pair(newBarrierProducers, newBarrierConsumers)));
            }
        }

        for (auto cons = consumers.begin(); cons != consumers.end(); cons++) {
            auto taskWaitItr = taskOpUpdateWaitMap.find(*cons);
            if (taskWaitItr != taskOpUpdateWaitMap.end()) {
                taskWaitItr->second.first.insert(barrierOp);
            } else {
                std::set<mlir::Operation*> newBarrierProducers{barrierOp};
                std::set<mlir::Operation*> newBarrierConsumers{};
                taskOpUpdateWaitMap.insert(
                        std::make_pair(*cons, std::make_pair(newBarrierProducers, newBarrierConsumers)));
            }
        }
    }
}

void TokenBasedBarrierScheduler::saveOriginalDependency() {
    configureBarrierOpUpdateWaitMapBackUp.clear();
    configureTaskOpUpdateWaitMapBackUp.clear();

    auto _barrierOps = to_small_vector(_func.getOps<VPURT::DeclareVirtualBarrierOp>());
    std::cout << "get initial barriers " << _barrierOps.size() << std::endl;
    for (auto& barrierOp : _barrierOps) {
        std::set<mlir::Operation*> producers{};
        std::set<mlir::Operation*> consumers{};

        for (auto* userOp : barrierOp->getUsers()) {
            std::cout << "get Users " << std::endl;
            auto opEffects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(userOp);

            VPUX_THROW_WHEN(opEffects == nullptr,
                            "Barrier '{0}' is used by Operation '{1}' without MemoryEffects interface",
                            barrierOp->getLoc(), userOp->getName());

            using MemEffect = mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>;

            SmallVector<MemEffect> valEffects;

            opEffects.getEffectsOnValue(barrierOp.barrier(), valEffects);

            VPUX_THROW_WHEN(
                    valEffects.size() != 1,
                    "Barrier '{0}' must have exactly 1 MemoryEffect per Operation, got '{1}' for Operation '{2}'",
                    barrierOp->getLoc(), valEffects.size(), userOp->getLoc());

            const auto& effect = valEffects.front();

            std::cout << "detect producers and consumers " << std::endl;

            if (effect.getEffect() == mlir::MemoryEffects::Write::get()) {
                auto task = mlir::dyn_cast<VPURT::TaskOp>(userOp);
                if (task == nullptr) {
                    exit(0);
                }
                // Logger::global().error("Task with scheduling number {0}", task->getAttr("SchedulingNumber"));
                if (task.getExecutorKind() == VPU::ExecutorKind::NCE) {
                    producers.insert(userOp);
                } else if (task.getExecutorKind() == VPU::ExecutorKind::DMA_NN) {
                    producers.insert(userOp);
                } else if (task.getExecutorKind() == VPU::ExecutorKind::SHAVE_UPA) {
                    producers.insert(userOp);
                }
            } else if (effect.getEffect() == mlir::MemoryEffects::Read::get()) {
                auto task = mlir::dyn_cast<VPURT::TaskOp>(userOp);
                if (task == nullptr) {
                    exit(0);
                }
                // Logger::global().error("Task with scheduling number {0}", task->getAttr("SchedulingNumber"));
                if (task.getExecutorKind() == VPU::ExecutorKind::NCE) {
                    consumers.insert(userOp);
                } else if (task.getExecutorKind() == VPU::ExecutorKind::DMA_NN) {
                    consumers.insert(userOp);
                } else if (task.getExecutorKind() == VPU::ExecutorKind::SHAVE_UPA) {
                    consumers.insert(userOp);
                }
            } else {
                VPUX_THROW("Barrier '{0}' has unsupported Effect in Operation '{1}'", barrierOp->getLoc(),
                           userOp->getLoc());
            }

            std::cout << "finish" << std::endl;
        }
        configureBarrierOpUpdateWaitMapBackUp.insert(std::make_pair(barrierOp, std::make_pair(producers, consumers)));
    }

    getTaskOpUpdateWaitMap(configureBarrierOpUpdateWaitMapBackUp, configureTaskOpUpdateWaitMapBackUp);

    _func->walk([](VPURT::TaskOp op) {
        op.updateBarriersMutable().clear();
        op.waitBarriersMutable().clear();
    });

    _func->walk([&](VPURT::DeclareVirtualBarrierOp op) {
        op->dropAllUses();
        op.erase();
    });

    std::cout << "Removed all declare virtual barrier ops" << std::endl;
}

bool TokenBasedBarrierScheduler::isPathExist(mlir::Operation* a, mlir::Operation* b) {
    auto numa = a->getAttr("SchedulingNumber").cast<mlir::IntegerAttr>().getInt();
    auto numb = b->getAttr("SchedulingNumber").cast<mlir::IntegerAttr>().getInt();
    if (numa >= numb)
        return false;
    else {
        auto updateBarriers = configureTaskOpUpdateWaitMap[a].second;
        std::set<mlir::Operation*> consumers;
        for (auto iter = updateBarriers.begin(); iter != updateBarriers.end(); iter++) {
            auto barrierConsumers = configureBarrierOpUpdateWaitMap[*iter].second;
            consumers.insert(barrierConsumers.begin(), barrierConsumers.end());
        }

        if (std::find(consumers.begin(), consumers.end(), b) != consumers.end())
            return true;
        else {
            for (auto consumer = consumers.begin(); consumer != consumers.end(); consumer++) {
                if (isPathExist(*consumer, b))
                    return true;
            }
        }
        return false;
    }
}

void TokenBasedBarrierScheduler::removeRedundantDependency() {
    for (auto iter = configureBarrierOpUpdateWaitMap.begin(); iter != configureBarrierOpUpdateWaitMap.end(); iter++) {
        // producers
        auto producers = (*iter).second.first;
        for (auto prod = producers.begin(); prod != producers.end();) {
            auto prod1 = prod;
            prod1++;
            for (; prod1 != producers.end();) {
                if (isPathExist(*prod1, *prod)) {
                    auto removedIter = prod1;
                    prod1++;
                    producers.erase(removedIter);
                    // configureTaskOpUpdateWaitMap[*prod1].second.erase((*iter).first);
                } else if (isPathExist(*prod, *prod1)) {
                    auto removedIter = prod;
                    prod++;
                    producers.erase(removedIter);
                    // configureTaskOpUpdateWaitMap[*prod].second.erase((*iter).first);
                    break;
                } else
                    prod1++;
            }
            if (prod1 == producers.end())
                prod++;
        }
        (*iter).second.first = producers;

        // consumers
        auto consumers = (*iter).second.second;
        for (auto cons = consumers.begin(); cons != consumers.end();) {
            auto cons1 = cons;
            cons1++;
            for (; cons1 != consumers.end();) {
                if (isPathExist(*cons, *cons1)) {
                    auto removedIter = cons1;
                    cons1++;
                    consumers.erase(removedIter);
                    // configureTaskOpUpdateWaitMap[*cons1].first.erase((*iter).first);
                } else if (isPathExist(*cons1, *cons)) {
                    auto removedIter = cons;
                    cons++;
                    consumers.erase(removedIter);
                    // configureTaskOpUpdateWaitMap[*cons].first.erase((*iter).first);
                    break;
                } else
                    cons1++;
            }
            if (cons1 == consumers.end())
                cons++;
        }
        (*iter).second.second = consumers;
    }
}

void TokenBasedBarrierScheduler::removeRedundantBarrier() {
    for (auto iter = configureBarrierOpUpdateWaitMap.begin(); iter != configureBarrierOpUpdateWaitMap.end(); iter++) {
        auto consumers = (*iter).second.second;
        auto iter1 = iter;
        iter1++;
        for (; iter1 != configureBarrierOpUpdateWaitMap.end();) {
            auto consumers1 = (*iter1).second.second;
            if (consumers1 == consumers) {
                Logger::global().error("found barrier {0} and {1} have same consumers", (*iter).first->getAttr("id"),
                                       (*iter1).first->getAttr("id"));
                auto producers = (*iter1).second.first;
                // auto mergedProducers = (*iter).second.first
                for (auto& task : producers) {
                    (*iter).second.first.insert(task);
                }
                auto removedIter = iter1;
                iter1++;
                (*removedIter).first->dropAllUses();
                (*removedIter).first->erase();
                configureBarrierOpUpdateWaitMap.erase(removedIter);
            } else
                iter1++;
        }
    }
}

void TokenBasedBarrierScheduler::reorderIR() {
    // reorder barrier by id
    mlir::Operation* preBarrier = nullptr;
    for (auto iter = configureBarrierOpUpdateWaitMap.begin(); iter != configureBarrierOpUpdateWaitMap.end(); iter++) {
        auto curBarrier = (*iter).first;
        if (preBarrier) {
            curBarrier->moveAfter(preBarrier);
        }
        preBarrier = curBarrier;
    }

    // reorder task by scheduling number
    mlir::Operation* preTask = nullptr;
    for (auto iter = configureTaskOpUpdateWaitMap.begin(); iter != configureTaskOpUpdateWaitMap.end(); iter++) {
        auto curTask = (*iter).first;
        if (preTask) {
            curTask->moveAfter(preTask);
        }
        preTask = curTask;
    }
}

size_t TokenBasedBarrierScheduler::schedule() {
    bool success = false;
    size_t btask_count = 0UL;

    for (; !success && (barrierCount_ >= 1UL); --barrierCount_) {
        // last associated barrier task associated with index //
        barrier_association_table_t barrier_association;

        Logger::global().error("STEP-0: Initialize the association table");
        for (size_t barrierId = 1; barrierId <= barrierCount_; barrierId++) {
            auto bitr =
                    barrier_association.insert(std::make_pair(barrierId, barrier_transition_structure_t(_func, *this)));
            barrier_transition_structure_t& bstructure = (bitr.first)->second;
            bstructure.init();
        }
        {
            Logger::global().error("STEP-1: run the scheduler");
            BarrierScheduleGenerator scheduler_begin(_ctx, _func, barrierCount_, slotCount_,
                                                     configureTaskOpUpdateWaitMapBackUp,
                                                     configureBarrierOpUpdateWaitMapBackUp);
            BarrierScheduleGenerator scheduler_end(_ctx, _func);
            size_t scheduling_number = 0UL;

            for (; scheduler_begin != scheduler_end; ++scheduler_begin) {
                const BarrierScheduleGenerator::schedule_info_t& sinfo = *scheduler_begin;

                Logger::global().error(
                        "Getting the schedule information sinfo from the Barrier_Schedule_Generator class");
                Logger::global().error("The time is {0}, the Operation is {1} end time is {1}", sinfo.schedule_time_,
                                       FeasibleScheduleGenerator::getUniqueID(sinfo.op_));
                Logger::global().error("The barrier index is {0}, , the slot cout is {1}", sinfo.barrier_index_,
                                       sinfo.slot_count_);

                Logger::global().error(
                        "Now looking up the barrier association table for the barrier {0} and getting the "
                        "barrier association table",
                        sinfo.barrier_index_);

                auto bitr = barrier_association.find(sinfo.barrier_index_);
                assert(bitr != barrier_association.end());
                barrier_transition_structure_t& bstructure = bitr->second;

                // Set scheduling number
                Logger::global().error("Assigning scheduling number {0} to the Operation {1} ", scheduling_number,
                                       FeasibleScheduleGenerator::getUniqueID(sinfo.op_));
                sinfo.op_->setAttr(schedulingNumberAttrName, getIntAttr(_ctx, scheduling_number));

                scheduling_number++;
                // STEP-2: update barrier structure invariant //
                bool new_barrier_task_created = bstructure.process_next_scheduled_op(sinfo, builder);

                if (new_barrier_task_created) {
                    ++btask_count;
                }
            }
        }

        // STEP-2.5: process trailing barrier control structures //
        {
            for (auto bitr = barrier_association.begin(); bitr != barrier_association.end(); ++bitr) {
                barrier_transition_structure_t& bstruct = bitr->second;
                bstruct.close_barrier_producer_list();
            }
        }

        // update,wait
        for (auto& barrier : configureBarrierOpUpdateWaitMap) {
            Logger::global().error("Barrier ID {0} has the following producers", barrier.first->getAttr("id"));
            for (auto op : barrier.second.first)
                Logger::global().error("producer Op with ID {0} to barrier {1}",
                                       FeasibleScheduleGenerator::getUniqueID(op), barrier.first->getAttr("id"));
        }

        for (auto& barrier : configureBarrierOpUpdateWaitMap) {
            Logger::global().error("Barrier ID {0} has the following consumers", barrier.first->getAttr("id"));
            for (auto op : barrier.second.second)
                Logger::global().error("consumer Op with ID {0} to barrier {1}",
                                       FeasibleScheduleGenerator::getUniqueID(op), barrier.first->getAttr("id"));
        }

        std::cout << "Done scheduling" << std::endl;

        getTaskOpUpdateWaitMap(configureBarrierOpUpdateWaitMap, configureTaskOpUpdateWaitMap);

        std::cout << "Done creating configureTaskOpUpdateWaitMap" << std::endl;

        removeRedundantDependency();
        removeRedundantBarrier();

        for (auto itr = configureBarrierOpUpdateWaitMap.begin(); itr != configureBarrierOpUpdateWaitMap.end();) {
            if (itr->second.second.empty()) {
                Logger::global().error("Earsing virtual Barrier ID {0} as it has no producers",
                                       itr->first->getAttr("id"));
                (*itr).first->dropAllUses();
                (*itr).first->erase();
                itr = configureBarrierOpUpdateWaitMap.erase(itr);
            } else {
                ++itr;
            }
        }

        // run simulation
        RuntimeSimulator simulator(_ctx, _func, _log, _numDmaEngines, 8);

        for (const auto& p : configureBarrierOpUpdateWaitMap) {
            auto barrierOp = mlir::dyn_cast_or_null<VPURT::DeclareVirtualBarrierOp>(p.first);
            Logger::global().error("Virtual Barrier ID {0} has {1} consumers", barrierOp->getAttr("id"),
                                   p.second.second.size());
        }

        for (const auto& p : configureBarrierOpUpdateWaitMap) {
            auto barrierOp = mlir::dyn_cast_or_null<VPURT::DeclareVirtualBarrierOp>(p.first);
            Logger::global().error("Virtual Barrier ID {0} has {1} producers", barrierOp->getAttr("id"),
                                   p.second.first.size());
        }

        for (const auto& p : configureBarrierOpUpdateWaitMap) {
            auto barrierOp = mlir::dyn_cast_or_null<VPURT::DeclareVirtualBarrierOp>(p.first);
            for (auto* user : p.second.first) {
                auto taskOp = mlir::dyn_cast_or_null<VPURT::TaskOp>(user);
                assert(taskOp != NULL);
                assert(barrierOp.barrier() != NULL);
                Logger::global().error("Adding Barrier ID {0} as an update barrier for operation {1}",
                                       barrierOp->getAttr("id"), FeasibleScheduleGenerator::getUniqueID(user));
                taskOp.updateBarriersMutable().append(barrierOp.barrier());
            }
        }

        for (const auto& p : configureBarrierOpUpdateWaitMap) {
            auto barrierOp = mlir::dyn_cast_or_null<VPURT::DeclareVirtualBarrierOp>(p.first);
            for (auto* user : p.second.second) {
                auto taskOp = mlir::dyn_cast_or_null<VPURT::TaskOp>(user);
                assert(taskOp != NULL);
                assert(barrierOp.barrier() != NULL);
                Logger::global().error("Adding Barrier ID {0} as an wait barrier for operation {1}",
                                       barrierOp->getAttr("id"), FeasibleScheduleGenerator::getUniqueID(user));
                taskOp.waitBarriersMutable().append(barrierOp.barrier());
            }
        }

        // success = true;
        success = simulator.assignPhysicalIDs();

        if (barrierCount_ == 4)
            success = false;

        if (!success) {
            _func->walk([](VPURT::TaskOp op) {
                op.updateBarriersMutable().clear();
                op.waitBarriersMutable().clear();
            });
            _func->walk([&](VPURT::DeclareVirtualBarrierOp op) {
                Logger::global().error("Erasing Barrier ID {0} ", op->getAttr("id"));
                op->dropAllUses();
                op.erase();
            });
            configureBarrierOpUpdateWaitMap.clear();
            configureTaskOpUpdateWaitMap.clear();
        }

        std::cout << "Barrier simualtion result is " << success << " with upperbound " << barrierCount_ << std::endl;
    }

    reorderIR();

    return btask_count;
}