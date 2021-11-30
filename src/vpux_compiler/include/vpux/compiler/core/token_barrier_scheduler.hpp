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

#pragma once

//#include "vpux/compiler/core/deps_info.hpp"
#include "vpux/compiler/core/mem_live_range_info.hpp"
#include "vpux/compiler/utils/linear_scan.hpp"

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/DenseSet.h>

#include "vpux/compiler/core/barrier_schedule_generator.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux {

class TokenBasedBarrierScheduler {
public:
    explicit TokenBasedBarrierScheduler(mlir::MLIRContext* ctx, mlir::FuncOp func, int64_t numBarriers, int64_t slotCount, Logger log);

    // typedef BarrierScheduleGenerator scheduler_t;
    typedef size_t schedule_time_t;
    std::unordered_map<mlir::Operation*, std::set<mlir::Operation*>> configureBarrierOpUpdateMap;
    std::unordered_map<mlir::Operation*, std::set<mlir::Operation*>> configureBarrierOpWaitMap;

    // One transition structure for each physical barrier //
    // TODO John: Move barrier_transition_structure_t to another file
    class barrier_transition_structure_t {
    public:

        //Outer class
        TokenBasedBarrierScheduler& tokenBasedBarrierScheduler_;
        
        
        typedef std::set<mlir::Operation*> producers_t;
        typedef typename producers_t::const_iterator producer_iterator_t;

        barrier_transition_structure_t(mlir::FuncOp func, TokenBasedBarrierScheduler& tokenBasedBarrierScheduler, schedule_time_t time = std::numeric_limits<schedule_time_t>::max())
                : _func(func), tokenBasedBarrierScheduler_(tokenBasedBarrierScheduler), time_(time), producers_()  {

            Logger::global().error("Initialising a new barrier_transition_structure");
        }

        void init() {
            time_ = std::numeric_limits<schedule_time_t>::max();
            prev_barrier_task_ = NULL;  // Initalize to Op 0 ?
            curr_barrier_task_ = NULL;
            producers_.clear();
        }

        // returns true if this call has resulted in creating a new barrier
        // task (also means there is a temporal change) //
        bool process_next_scheduled_op(const BarrierScheduleGenerator::schedule_info_t& sinfo) {
            schedule_time_t curr_time = sinfo.schedule_time_;
            bool created_new_barrier_task = false;

            Logger::global().error("The scheduled time is {0}, the op is {1} the barrier index is {2}  the slot cout is {3}", sinfo.schedule_time_, FeasibleScheduleGenerator::getUniqueID(sinfo.op_), sinfo.barrier_index_, sinfo.slot_count_);

            Logger::global().error("The global time is {0}", time_);
            Logger::global().error("The current time is {0}", curr_time);
            
            if (time_ != curr_time) {
                Logger::global().error("CASE-1: temporal transition happened, create a new barrier task -  maintain_invariant_temporal_change");
                // CASE-1: temporal transition happened //
                created_new_barrier_task = true;
                maintain_invariant_temporal_change(sinfo);
                time_ = curr_time;
            } else {
                // CASE-2: trival case //
                Logger::global().error("CASE-2: trival case - add_scheduled_op_to_producer_list");
                add_scheduled_op_to_producer_list(sinfo);
              }
              return created_new_barrier_task;
        }

          void close_barrier_producer_list() {
          if (curr_barrier_task_ == NULL) { return; }
          process_current_barrier_producer_list_close_event(curr_barrier_task_,
              prev_barrier_task_);
        }
        

    private:

         void maintain_invariant_temporal_change(const BarrierScheduleGenerator::schedule_info_t& sinfo) {

          Logger::global().error("Calling maintain_invariant_temporal_change()");
          Logger::global().error("The scheduled time is {0}, the op is {1} the barrier index is {2}  the slot cout is {3}", sinfo.schedule_time_, FeasibleScheduleGenerator::getUniqueID(sinfo.op_), sinfo.barrier_index_, sinfo.slot_count_);
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
        
          bop_curr_new = create_new_barrier_task(sinfo);

         assert(bop_curr_new != bop_end);
         //assert(is_barrier_task(bop_curr_new));

          // STEP-1 //
           std::cout << "bop_curr = " << bop_curr << " bop_end = " << bop_end << std::endl;
          if (bop_curr != bop_end) {
            Logger::global().error("The ID of barrier bop_curr is {0}", bop_curr->getAttr("id"));
            process_current_barrier_producer_list_close_event(bop_curr, bop_prev);
          }

          // STEP-2 //
          prev_barrier_task_ = curr_barrier_task_;
          curr_barrier_task_ = bop_curr_new;
         producers_.clear();
          add_scheduled_op_to_producer_list(sinfo);

         }

        // Maintain the invariant given that that we are closing the producer
        // list of the current barrier.
        inline void process_current_barrier_producer_list_close_event(mlir::Operation* bop_curr, mlir::Operation* bop_prev) {

            Logger::global().error("Process current barrier producer list close event");
            
            mlir::Operation* bop_end = NULL;
            assert(bop_curr != bop_end);

            // Get the barrier object for the three barrier tasks //
            mlir::Operation* b_prev = NULL;

            if (bop_prev != bop_end) {
                b_prev = bop_prev;
            }

            Logger::global().error("The ID of barrier b_curr is {0}", bop_curr->getAttr("id"));


            for (producer_iterator_t producer=producers_.begin(); producer!=producers_.end(); ++producer) {

                mlir::Operation* source = *producer;

    
                // STEP-1.2 (a): producers //
                auto barrierProducersItr = tokenBasedBarrierScheduler_.configureBarrierOpUpdateMap.find(bop_curr);

                if(barrierProducersItr !=  tokenBasedBarrierScheduler_.configureBarrierOpUpdateMap.end())
                {
                    Logger::global().error("Adding producer Op with ID {0} to barrier {1}", FeasibleScheduleGenerator::getUniqueID(source),  bop_curr->getAttr("id"));
                    barrierProducersItr->second.insert(source);
                }
                else
                    VPUX_THROW("Not found");
                

                // STEP-1.2 (b): consumers //
                auto barrierConsumersItr = tokenBasedBarrierScheduler_.configureBarrierOpWaitMap.find(bop_curr);
                
                if(barrierConsumersItr !=  tokenBasedBarrierScheduler_.configureBarrierOpWaitMap.end())
                {
                    auto opConsumers = FeasibleScheduleGenerator::getConsumerOps(source);
                    for (auto consumer = opConsumers.begin(); consumer != opConsumers.end(); ++consumer) {
                        Logger::global().error("STEP-1.2 Adding consumer Op with ID {0} to barrier {1}", FeasibleScheduleGenerator::getUniqueID(*consumer),  bop_curr->getAttr("id"));
                        barrierConsumersItr->second.insert(*consumer);
                    }
                }
                else 
                     VPUX_THROW("Not found");
                
                // STEP-1.3 //
                if (b_prev) {
                    auto barrierConsumersItr = tokenBasedBarrierScheduler_.configureBarrierOpWaitMap.find(b_prev);
                    if(barrierConsumersItr !=  tokenBasedBarrierScheduler_.configureBarrierOpWaitMap.end())
                    {
                        Logger::global().error("STEP-1.3 Adding consumer Op with ID {0} to barrier {1}", FeasibleScheduleGenerator::getUniqueID(source),  b_prev->getAttr("id"));
                        barrierConsumersItr->second.insert(source);
                    }
                    else 
                        VPUX_THROW("Not found");
                }
            } // foreach producer //
        }

         void add_scheduled_op_to_producer_list(const BarrierScheduleGenerator::schedule_info_t& sinfo) {
            auto scheduled_op = sinfo.op_;

            Logger::global().error("Adding op {0} to the producer list of the barrier transition structure", curr_barrier_task_->getAttr("id"));
            producers_.insert(scheduled_op);
        }

        mlir::Operation* create_new_barrier_task(const BarrierScheduleGenerator::schedule_info_t& sinfo) {

             Logger::global().error("CREATING A NEW BARRIER TASK");

             static size_t barrier_task_id=0UL;

             mlir::OpBuilder builder(_func.getBody());

             builder.setInsertionPoint(sinfo.op_);
             mlir::Operation* newBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(sinfo.op_->getLoc(), barrier_task_id); //Neds to be virtual.

             std::set<mlir::Operation*> newBarrierProducers{};
             std::set<mlir::Operation*> newBarrierConsumers{};            
             tokenBasedBarrierScheduler_.configureBarrierOpUpdateMap.insert(std::make_pair(newBarrier, newBarrierProducers));
             tokenBasedBarrierScheduler_.configureBarrierOpWaitMap.insert(std::make_pair(newBarrier, newBarrierConsumers));

             Logger::global().error("Created a new barrier task with barrier ID {0} after OP id is {1}", barrier_task_id, FeasibleScheduleGenerator::getUniqueID(sinfo.op_));
             barrier_task_id++;
             return newBarrier;

         }

        schedule_time_t time_;
        mlir::Operation* curr_barrier_task_;
        mlir::Operation* prev_barrier_task_;
        producers_t producers_;
        mlir::FuncOp _func;

    };  // class barrier_transition_structure_t //

    size_t schedule();

private:
    typedef std::unordered_map<size_t, barrier_transition_structure_t> barrier_association_table_t;

    mlir::MLIRContext* _ctx;
    mlir::FuncOp _func;
    size_t barrierCount_;
    size_t slotCount_;
    
};

}  // namespace vpux
