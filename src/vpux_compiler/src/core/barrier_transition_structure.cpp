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

using namespace vpux::VPURT;

FeasibleBarrierScheduler::barrierTransitionStructure::barrierTransitionStructure(
        FeasibleBarrierScheduler& feasibleBarrierScheduler, size_t time)
        : _feasibleBarrierScheduler(feasibleBarrierScheduler), time_(time), producers_() {
    _feasibleBarrierScheduler._log.trace("Initialising a new barrier_transition_structure");
}

void FeasibleBarrierScheduler::barrierTransitionStructure::init() {
    time_ = std::numeric_limits<size_t>::max();
    prev_barrier_task_ = NULL;
    curr_barrier_task_ = NULL;
    producers_.clear();
}

bool FeasibleBarrierScheduler::barrierTransitionStructure::processNextScheduledTask(const ScheduledOpInfo& sinfo,
                                                                                     mlir::OpBuilder& builder) {
    size_t curr_time = sinfo._scheduleTime;
    bool created_new_barrier_task = false;

    _feasibleBarrierScheduler._log.trace("The scheduled time is {0}, the op is {1} the barrier index is {2}  the slot cout is {3}",
                           sinfo._scheduleTime, FeasibleBarrierScheduler::getUniqueID(sinfo._op), sinfo._barrierIndex,
                           sinfo._producerSlotCount);

    _feasibleBarrierScheduler._log.trace("The global time is {0}", time_);
    _feasibleBarrierScheduler._log.trace("The current time is {0}", curr_time);

    if (time_ != curr_time) {
        _feasibleBarrierScheduler._log.trace("CASE-1: temporal transition happened, create a new barrier task -  "
                                             "maintainInvariantTemporalChange");
        // CASE-1: temporal transition happened //
        created_new_barrier_task = true;
        maintainInvariantTemporalChange(sinfo, builder);
        time_ = curr_time;
    } else {
        // CASE-2: trival case //
        _feasibleBarrierScheduler._log.trace("CASE-2: trival case - addScheduledOpToProducerList");
        addScheduledOpToProducerList(sinfo);
    }
    return created_new_barrier_task;
}

void FeasibleBarrierScheduler::barrierTransitionStructure::closeBarrierProducerList() {
    if (curr_barrier_task_ == NULL) {
        return;
    }
    processCurrentBarrierProducerListCloseEvent(curr_barrier_task_, prev_barrier_task_);
}

inline void FeasibleBarrierScheduler::barrierTransitionStructure::processCurrentBarrierProducerListCloseEvent(
        mlir::Operation* bop_curr, mlir::Operation* bop_prev) {
    _feasibleBarrierScheduler._log.trace("Process current barrier producer list close event");

    mlir::Operation* bop_end = NULL;
    assert(bop_curr != bop_end);

    // Get the barrier object for the three barrier tasks //
    mlir::Operation* b_prev = NULL;

    if (bop_prev != bop_end) {
        b_prev = bop_prev;
    }

    _feasibleBarrierScheduler._log.trace("The ID of barrier b_curr is {0}", bop_curr->getAttr("id"));

    for (producer_iterator_t producer = producers_.begin(); producer != producers_.end(); ++producer) {
        mlir::Operation* source = *producer;

        // STEP-1.2 (a): producers //
        auto barrierProducersItr = _feasibleBarrierScheduler.configureBarrierOpUpdateWaitMap.find(bop_curr);

        if (barrierProducersItr != _feasibleBarrierScheduler.configureBarrierOpUpdateWaitMap.end()) {
            _feasibleBarrierScheduler._log.trace("Adding producer Op with ID {0} to barrier {1}",
                                                 FeasibleBarrierScheduler::getUniqueID(source),
                                                 bop_curr->getAttr("id"));
            barrierProducersItr->second.first.insert(source);
        } else
            VPUX_THROW("Not found");

        // STEP-1.2 (b): consumers //
        auto barrierConsumersItr = _feasibleBarrierScheduler.configureBarrierOpUpdateWaitMap.find(bop_curr);

        if (barrierConsumersItr != _feasibleBarrierScheduler.configureBarrierOpUpdateWaitMap.end()) {
            auto opConsumers = _feasibleBarrierScheduler.getConsumerOps(source);

            for (auto consumer = opConsumers.begin(); consumer != opConsumers.end(); ++consumer) {
                _feasibleBarrierScheduler._log.trace("STEP-1.2 Adding consumer Op with ID {0} to barrier {1}",
                                                     FeasibleBarrierScheduler::getUniqueID(*consumer),
                                                     bop_curr->getAttr("id"));
                barrierConsumersItr->second.second.insert(*consumer);
            }
        } else
            VPUX_THROW("Not found");

        // STEP-1.3 //
        if (b_prev) {
            auto barrierConsumersItr = _feasibleBarrierScheduler.configureBarrierOpUpdateWaitMap.find(b_prev);
            if (barrierConsumersItr != _feasibleBarrierScheduler.configureBarrierOpUpdateWaitMap.end()) {
                _feasibleBarrierScheduler._log.trace("STEP-1.3 Adding consumer Op with ID {0} to barrier {1}",
                                                     FeasibleBarrierScheduler::getUniqueID(source),
                                                     b_prev->getAttr("id"));
                barrierConsumersItr->second.second.insert(source);
            } else
                VPUX_THROW("Not found");
        }
    }  // foreach producer //
}

void FeasibleBarrierScheduler::barrierTransitionStructure::maintainInvariantTemporalChange(const ScheduledOpInfo& sinfo,
                                                                                           mlir::OpBuilder& builder) {
    _feasibleBarrierScheduler._log.trace("Calling maintainInvariantTemporalChange()");
    _feasibleBarrierScheduler._log.trace("The scheduled time is {0}, the op is {1} the barrier index is {2}  the slot cout is {3}",
                           sinfo._scheduleTime, FeasibleBarrierScheduler::getUniqueID(sinfo._op), sinfo._barrierIndex,
                           sinfo._producerSlotCount);
    //              B_prev
    // curr_state : Prod_list={p_0, p_1, ... p_n}-->B_curr
    // event: Prod_list={q_0}->B_curr_new
    //
    // scheduler says it want to associate both B_old and B_new to the
    // same physical barrier.
    //
    // Restore Invariant:
    // STEP-1.1: create a new barrier task (B_new).
    // STEP-1.2: update B_curr
    //        a. producers: B_curr is now closed so update its producers
    //        b. consumers: for each (p_i, u) \in P_old x V
    //                      add u to the consumer list of B_old
    // STEP-1.3: update B_prev
    //           consumers: add p_i \in P_old to the consumer list of
    //                      B_prev. This is because B_prev and B_curr
    //                      are associated with same physical barrier.
    // STEP-2: B_prev = B_curr , B_curr = B_curr_new , Prod_list ={q0}
    mlir::Operation* bop_prev = prev_barrier_task_;
    mlir::Operation* bop_curr = curr_barrier_task_;
    mlir::Operation* bop_end = NULL;
    mlir::Operation* bop_curr_new = bop_end;

    bop_curr_new = createNewBarrierTask(sinfo, builder);

    assert(bop_curr_new != bop_end);
    // assert(is_barrier_task(bop_curr_new));

    // STEP-1 //
    if (bop_curr != bop_end) {
        _feasibleBarrierScheduler._log.trace("The ID of barrier bop_curr is {0}", bop_curr->getAttr("id"));
        processCurrentBarrierProducerListCloseEvent(bop_curr, bop_prev);
    }

    // STEP-2 //
    prev_barrier_task_ = curr_barrier_task_;
    curr_barrier_task_ = bop_curr_new;
    producers_.clear();
    addScheduledOpToProducerList(sinfo);
}

void FeasibleBarrierScheduler::barrierTransitionStructure::addScheduledOpToProducerList(
        const ScheduledOpInfo& sinfo) {
    auto scheduled_op = sinfo._op;

    _feasibleBarrierScheduler._log.trace("Adding op {0} to the producer list of the barrier transition structure",
                                         curr_barrier_task_->getAttr("id"));
    producers_.insert(scheduled_op);
}

mlir::Operation* FeasibleBarrierScheduler::barrierTransitionStructure::createNewBarrierTask(
        const ScheduledOpInfo& sinfo, mlir::OpBuilder& builder) {
    _feasibleBarrierScheduler._log.trace("CREATING A NEW BARRIER TASK");

    static size_t barrier_task_id = 1UL;

    auto newBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(sinfo._op->getLoc());
    newBarrier->setAttr(virtualIdAttrName, getIntAttr(newBarrier->getContext(), barrier_task_id));

    std::set<mlir::Operation*> newBarrierProducers{};
    std::set<mlir::Operation*> newBarrierConsumers{};
    _feasibleBarrierScheduler.configureBarrierOpUpdateWaitMap.insert(
            std::make_pair(newBarrier, std::make_pair(newBarrierProducers, newBarrierConsumers)));

    _feasibleBarrierScheduler._log.trace("Created a new barrier task with barrier ID {0} after OP id is {1}", barrier_task_id,
                           FeasibleBarrierScheduler::getUniqueID(sinfo._op));
    barrier_task_id++;
    return newBarrier;
}