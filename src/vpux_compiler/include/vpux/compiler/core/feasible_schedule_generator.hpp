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

//#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"
#include "vpux/compiler/core/barrier_resource_state.hpp"
#include "vpux/compiler/core/op_resource_state.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"

namespace vpux {

struct task_operation_comparator_by_schedule_time_t {
    bool operator()(mlir::Operation* op1, mlir::Operation* op2) const {
        int64_t schedulingNumber1 = checked_cast<int64_t>(
                mlir::dyn_cast<VPURT::TaskOp>(op1)->getAttr("SchedulingNumber").cast<mlir::IntegerAttr>().getInt());
        int64_t schedulingNumber2 = checked_cast<int64_t>(
                mlir::dyn_cast<VPURT::TaskOp>(op2)->getAttr("SchedulingNumber").cast<mlir::IntegerAttr>().getInt());

        return schedulingNumber1 < schedulingNumber2;
    }
};
struct operation_comparator_t {
    bool operator()(mlir::Operation* op1, mlir::Operation* op2) const {
        int64_t uniqueId1 = checked_cast<int64_t>(mlir::dyn_cast<VPURT::DeclareVirtualBarrierOp>(op1)
                                                          ->getAttr(virtualIdAttrName)
                                                          .cast<mlir::IntegerAttr>()
                                                          .getInt());
        int64_t uniqueId2 = checked_cast<int64_t>(mlir::dyn_cast<VPURT::DeclareVirtualBarrierOp>(op2)
                                                          ->getAttr(virtualIdAttrName)
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
struct op_resource_state_t;
class FeasibleScheduleGenerator {
public:
    typedef size_t schedule_time_t;
    struct heap_element_t {
        heap_element_t(mlir::Operation* op = NULL, schedule_time_t t = 0UL): op_(op), time_(t) {
        }

        mlir::Operation* op_;
        schedule_time_t time_;
    };  // struct heap_element_t //

    struct min_heap_ordering_t {
        bool operator()(const heap_element_t& a, const heap_element_t& b) {
            return a.time_ > b.time_;
        }
    };  // struct min_heap_ordering_t //

    using delay_t = size_t;
    using schedulable_ops_t = std::list<mlir::Operation*>;
    typedef typename schedulable_ops_t::iterator schedulable_ops_iterator_t;
    using processed_ops_t = std::set<mlir::Operation*>;
    using schedule_heap_t = std::vector<heap_element_t>;
    using operation_in_degree_t = std::map<mlir::Operation*, size_t, task_operation_comparator_t>;
    using priority_map_t = std::map<mlir::Operation*, size_t, task_operation_comparator_t>;
    using resource_utility_map_t = std::unordered_map<mlir::Operation*, unsigned>;
    resource_utility_map_t resource_utility_map_;

    FeasibleScheduleGenerator(
            mlir::MLIRContext* ctx, mlir::FuncOp func, const resource_state_t& rstate,
            std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>>&
                    taskOpUpdateWaitMap,
            std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>>&
                    barrierOpUpdateWaitMap);
    FeasibleScheduleGenerator(mlir::MLIRContext* ctx, mlir::FuncOp func);

    bool operator==(const FeasibleScheduleGenerator& o) const;
    bool reached_end() const;
    void operator++();
    bool next_schedulable_operation();
    mlir::Operation*& operator*();  // should be const ?
    size_t current_time() const;
    const resource_state_t& resource_state() const;
    void getAllBarriersProducersAndConsumers();
    bool init(const resource_state_t& upper_bound);
    void init_resource_state(const resource_state_t& start_state);
    void compute_op_indegree(operation_in_degree_t& in_degree);
    void add_to_candidate_set(mlir::Operation* op);
    void compute_operation_priorities();
    schedulable_ops_iterator_t find_schedulable_op();
    void create_resource_utility_table_for_barrier_scheduling();
    bool doesOpRunOnNCE(mlir::Operation* op);
    unsigned countProducerConsumerTasks(mlir::Operation* op);
    SmallVector<mlir::Operation*> getConsumerOps(mlir::Operation* op);
    static std::string printOpType(VPURT::TaskOp taskOp);
    void printInfo(mlir::FuncOp func);
    bool is_valid_op(schedulable_ops_iterator_t itr) const;
    void pushToHeap(const heap_element_t& elem);
    heap_element_t popFromHeap();
    void add_outgoing_operations_to_candidate_list(mlir::Operation* op);
    void assignUniqueIds();
    static mlir::IntegerAttr getUniqueID(mlir::Operation* op);

protected:
    mlir::MLIRContext* _ctx;
    mlir::FuncOp _func;
    operation_in_degree_t in_degree_;
    schedule_heap_t heap_;
    schedule_time_t current_time_;
    schedulable_ops_t candidates_;
    resource_state_t resource_state_;
    min_heap_ordering_t heap_ordering_;
    mlir::Operation* schedulable_op_;
    processed_ops_t processed_ops_;
    priority_map_t priority_;
    // outputs of the graph
    llvm::DenseSet<mlir::Operation*> _outputOps;
    // operation out-degree, number of outgoing edges
    std::map<mlir::Operation*, size_t> _outDegreeTable;

    // std::unordered_map<mlir::Operation*, size_t> _operationInDegree;
    // std::unordered_map<mlir::Operation*, size_t> _operationOutDegree;
    SmallVector<IERT::LayerOpInterface> _allTaskOps;
    SmallVector<VPURT::DeclareVirtualBarrierOp> _allBarrierOps;
    static std::map<mlir::Operation*, SmallVector<mlir::Operation*>> barrierProducersMap;
    static std::map<mlir::Operation*, SmallVector<mlir::Operation*>> barrierConsumersMap;
    std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>>
            _barrierOpUpdateWaitMap;
    std::map<mlir::Operation*, std::pair<std::set<mlir::Operation*>, std::set<mlir::Operation*>>>
            _taskOpUpdateWaitMap{};
};

}  // namespace vpux
