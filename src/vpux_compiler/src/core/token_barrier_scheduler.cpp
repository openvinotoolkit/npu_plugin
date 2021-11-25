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

//
// Constructor
//

TokenBasedBarrierScheduler::TokenBasedBarrierScheduler(mlir::MLIRContext* ctx, mlir::FuncOp func, int64_t numBarriers,
                                                       int64_t slotCount, Logger log)
        : _ctx(ctx), _func(func), barrierCount_(numBarriers), slotCount_(slotCount) {
}

size_t TokenBasedBarrierScheduler::schedule() {
    size_t btask_count = 0UL;

    // last associated barrier task associated with index //
    barrier_association_table_t barrier_association;

    Logger::global().error("STEP-0: Initialize the association table");
    for (size_t barrierId = 1; barrierId <= barrierCount_; barrierId++) {
        auto bitr = barrier_association.insert(std::make_pair(barrierId, barrier_transition_structure_t(*this)));
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
            scheduling_number++;
            // STEP-2: update barrier structure invariant //
            bool new_barrier_task_created = bstructure.process_next_scheduled_op(sinfo);

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

    for (auto& barrier : configureBarrierOpProducersMap) {
        Logger::global().error("Barrier ID {0} has the following producers", barrier.first->getAttr("id"));
        for (auto op : barrier.second)
            Logger::global().error("producer Op with ID {0} to barrier {1}", FeasibleScheduleGenerator::getUniqueID(op),
                                   barrier.first->getAttr("id"));
    }

    for (auto& barrier : configureBarrierOpConsumersMap) {
        Logger::global().error("Barrier ID {0} has the following consumers", barrier.first->getAttr("id"));
        for (auto op : barrier.second)
            Logger::global().error("consumer Op with ID {0} to barrier {1}", FeasibleScheduleGenerator::getUniqueID(op),
                                   barrier.first->getAttr("id"));
    }

    std::cout << "Done scheduling" << std::endl;

    _func->walk([](VPURT::DeclareVirtualBarrierOp op) {
        op.barrier().dropAllUses();
        op.erase();
    });

    for (const auto& p : configureBarrierOpProducersMap) {
        auto barrierOp = mlir::cast<VPURT::ConfigureBarrierOp>(p.first);
        for (auto* user : p.second) {
            auto taskOp = mlir::cast<VPURT::TaskOp>(user);
            Logger::global().error("Adding Barrier ID {0} as an update barrier for operation {1}", barrierOp->getAttr("id"),FeasibleScheduleGenerator::getUniqueID(user));
            taskOp.updateBarriersMutable().append(barrierOp.barrier());
        }
    }

    for (const auto& p : configureBarrierOpConsumersMap) {
        auto barrierOp = mlir::cast<VPURT::ConfigureBarrierOp>(p.first);
        for (auto* user : p.second) {
            auto taskOp = mlir::cast<VPURT::TaskOp>(user);
            Logger::global().error("Adding Barrier ID {0} as an wait barrier for operation {1}", barrierOp->getAttr("id"),FeasibleScheduleGenerator::getUniqueID(user));
            taskOp.waitBarriersMutable().append(barrierOp.barrier());
        }
    }

    std::cout << "Removed all declare virtual barrier ops" << std::endl;
}