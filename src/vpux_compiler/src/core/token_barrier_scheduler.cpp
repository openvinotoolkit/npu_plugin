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

TokenBasedBarrierScheduler::TokenBasedBarrierScheduler(mlir::MLIRContext* ctx, mlir::FuncOp func, int64_t numBarriers,
                                                       int64_t slotCount)
        : _ctx(ctx), _func(func), builder(_func.getBody()), barrierCount_(numBarriers), slotCount_(slotCount) {
}

size_t TokenBasedBarrierScheduler::schedule() {
    size_t btask_count = 0UL;

    // last associated barrier task associated with index //
    barrier_association_table_t barrier_association;

    Logger::global().error("STEP-0: Initialize the association table");
    for (size_t barrierId = 1; barrierId <= barrierCount_; barrierId++) {
        auto bitr = barrier_association.insert(std::make_pair(barrierId, barrier_transition_structure_t(_func, *this)));
        barrier_transition_structure_t& bstructure = (bitr.first)->second;
        bstructure.init();
    }
    {
        Logger::global().error("STEP-1: run the scheduler");
        BarrierScheduleGenerator scheduler_begin(_ctx, _func, barrierCount_, slotCount_);
        BarrierScheduleGenerator scheduler_end(_ctx, _func);
        size_t scheduling_number = 0UL;

        for (; scheduler_begin != scheduler_end; ++scheduler_begin) {
            const BarrierScheduleGenerator::schedule_info_t& sinfo = *scheduler_begin;

            Logger::global().error("Getting the schedule information sinfo from the Barrier_Schedule_Generator class");
            Logger::global().error("The time is {0}, the Operation is {1} end time is {1}", sinfo.schedule_time_,
                                   FeasibleScheduleGenerator::getUniqueID(sinfo.op_));
            Logger::global().error("The barrier index is {0}, , the slot cout is {1}", sinfo.barrier_index_,
                                   sinfo.slot_count_);

            Logger::global().error("Now looking up the barrier association table for the barrier {0} and getting the "
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
            Logger::global().error("producer Op with ID {0} to barrier {1}", FeasibleScheduleGenerator::getUniqueID(op),
                                   barrier.first->getAttr("id"));
    }

    for (auto& barrier : configureBarrierOpUpdateWaitMap) {
        Logger::global().error("Barrier ID {0} has the following consumers", barrier.first->getAttr("id"));
        for (auto op : barrier.second.second)
            Logger::global().error("consumer Op with ID {0} to barrier {1}", FeasibleScheduleGenerator::getUniqueID(op),
                                   barrier.first->getAttr("id"));
    }

    std::cout << "Done scheduling" << std::endl;

    // remove redundant barriers
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
                configureBarrierOpUpdateWaitMap.erase(removedIter);
            } else
                iter1++;
        }
    }

    for (auto& barrier : configureBarrierOpUpdateWaitMap) {
        Logger::global().error("Barrier ID {0} has the following producers", barrier.first->getAttr("id"));
        for (auto op : barrier.second.first)
            Logger::global().error("producer Op with ID {0} to barrier {1}", FeasibleScheduleGenerator::getUniqueID(op),
                                   barrier.first->getAttr("id"));
    }

    _func->walk([](VPURT::TaskOp op) {
        op.updateBarriersMutable().clear();
        op.waitBarriersMutable().clear();
    });

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

    for (auto itr = configureBarrierOpUpdateWaitMap.begin(); itr != configureBarrierOpUpdateWaitMap.end();) {
        if (itr->second.second.empty()) {
            Logger::global().error("Earsing virtual Barrier ID {0} as it has no producers", itr->first->getAttr("id"));
            itr = configureBarrierOpUpdateWaitMap.erase(itr);
        } else {
            ++itr;
        }
    }

    _func->walk([&](VPURT::DeclareVirtualBarrierOp op) {
        if (!configureBarrierOpUpdateWaitMap.count(op)) {
            op->dropAllUses();
            op.erase();
        }
    });

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

    std::cout << "Removed all declare virtual barrier ops" << std::endl;
    return btask_count;
}