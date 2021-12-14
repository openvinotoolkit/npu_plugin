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
    explicit TokenBasedBarrierScheduler(mlir::MLIRContext* ctx, mlir::FuncOp func, int64_t numBarriers,
                                        int64_t slotCount, Logger log);

    struct operation_comparator_t {
        bool operator()(mlir::Operation* op1, mlir::Operation* op2) const {
            int64_t uniqueId1 = checked_cast<int64_t>(mlir::dyn_cast<VPURT::DeclareVirtualBarrierOp>(op1)
                                                              ->getAttr("id")
                                                              .cast<mlir::IntegerAttr>()
                                                              .getInt());
            int64_t uniqueId2 = checked_cast<int64_t>(mlir::dyn_cast<VPURT::DeclareVirtualBarrierOp>(op2)
                                                              ->getAttr("id")
                                                              .cast<mlir::IntegerAttr>()
                                                              .getInt());

            return uniqueId1 < uniqueId2;
        }
    };

    struct task_operation_comparator_t {
        bool operator()(mlir::Operation* op1, mlir::Operation* op2) const {
            int64_t uniqueId1 = checked_cast<int64_t>(
                    mlir::dyn_cast<VPURT::TaskOp>(op1)->getAttr(uniqueIdAttrName).cast<mlir::IntegerAttr>().getInt());
            int64_t uniqueId2 = checked_cast<int64_t>(
                    mlir::dyn_cast<VPURT::TaskOp>(op2)->getAttr(uniqueIdAttrName).cast<mlir::IntegerAttr>().getInt());

            return uniqueId1 < uniqueId2;
        }
    };

    using schedule_time_t = size_t;

    /*Stores every barrier's associated update and wait operations*/
    std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*, task_operation_comparator_t>, std::set<mlir::Operation*, task_operation_comparator_t>>, operation_comparator_t> configureBarrierOpUpdateWaitMap;  // update,wait

    
    class barrierTransitionStructure {
    public:

        barrierTransitionStructure(mlir::FuncOp func, TokenBasedBarrierScheduler& tokenBasedBarrierScheduler,
                                       schedule_time_t time = std::numeric_limits<schedule_time_t>::max());

        void init();
        bool process_next_scheduled_op(const BarrierScheduleGenerator::schedule_info_t& sinfo, mlir::OpBuilder& builder);
        void close_barrier_producer_list();
        struct operation_comparator_t {
            bool operator()(mlir::Operation* op1, mlir::Operation* op2) const {
                int64_t uniqueId1 = checked_cast<int64_t>(mlir::dyn_cast<VPURT::TaskOp>(op1)
                                                                  ->getAttr(uniqueIdAttrName)
                                                                  .cast<mlir::IntegerAttr>()
                                                                  .getInt());
                int64_t uniqueId2 = checked_cast<int64_t>(mlir::dyn_cast<VPURT::TaskOp>(op2)
                                                                  ->getAttr(uniqueIdAttrName)
                                                                  .cast<mlir::IntegerAttr>()
                                                                  .getInt());

                return uniqueId1 < uniqueId2;
            }
        };

        using producers_t = std::set<mlir::Operation*, operation_comparator_t>;
        using producer_iterator_t = typename producers_t::const_iterator;

    private:
        void maintain_invariant_temporal_change(const BarrierScheduleGenerator::schedule_info_t& sinfo, mlir::OpBuilder& builder);
        inline void process_current_barrier_producer_list_close_event(mlir::Operation* bop_curr, mlir::Operation* bop_prev);
        void add_scheduled_op_to_producer_list(const BarrierScheduleGenerator::schedule_info_t& sinfo);
        mlir::Operation* create_new_barrier_task(const BarrierScheduleGenerator::schedule_info_t& sinfo, mlir::OpBuilder& builder);

        mlir::FuncOp _func;
        // Outer class
        TokenBasedBarrierScheduler& tokenBasedBarrierScheduler_;
        schedule_time_t time_;
        mlir::Operation* curr_barrier_task_;
        mlir::Operation* prev_barrier_task_;
        producers_t producers_;
    };  

    size_t schedule();
    bool isPathExist(mlir::Operation* a, mlir::Operation* b);

private:
    typedef std::unordered_map<size_t, barrierTransitionStructure> barrier_association_table_t;

    Logger _log;
    mlir::MLIRContext* _ctx;
    mlir::FuncOp _func;
    mlir::OpBuilder _builder;
    size_t _barrierCount;
    size_t _slotCount;
};

}  // namespace vpux
