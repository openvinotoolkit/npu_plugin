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

#include "vpux/compiler/core/barrier_resource_state.hpp"
#include "vpux/compiler/core/op_resource_state.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPURT/ops.hpp"

namespace vpux {

struct op_resource_state_t;
static constexpr StringLiteral uniqueIdAttrName = "uniqueId";
class FeasibleBarrierScheduler {
public:
    struct operation_comparator_t {
        bool operator()(mlir::Operation* op1, mlir::Operation* op2) const {
            int64_t uniqueId1 = checked_cast<int64_t>(
                    mlir::dyn_cast<VPURT::TaskOp>(op1)->getAttr(uniqueIdAttrName).cast<mlir::IntegerAttr>().getInt());
            int64_t uniqueId2 = checked_cast<int64_t>(
                    mlir::dyn_cast<VPURT::TaskOp>(op2)->getAttr(uniqueIdAttrName).cast<mlir::IntegerAttr>().getInt());

            return uniqueId1 < uniqueId2;
        }
    };
    struct HeapElement {
        HeapElement(mlir::Operation* op = NULL, schedule_time_t t = 0UL): op_(op), time_(t) {
        }

        mlir::Operation* op_;
        schedule_time_t time_;
    };

    struct MinHeapOrdering {
        bool operator()(const HeapElement& a, const HeapElement& b) {
            return a.time_ > b.time_;
        }
    };

    using delay_t = size_t;
    using schedulable_ops_t = std::list<mlir::Operation*>;
    using schedulable_ops_iterator_t = typename schedulable_ops_t::iterator;
    using processed_ops_t = std::set<mlir::Operation*>;
    using schedule_heap_t = std::vector<HeapElement>;
    using operation_in_degree_t = std::map<mlir::Operation*, size_t, operation_comparator_t>;
    using priority_map_t = std::map<mlir::Operation*, size_t, operation_comparator_t>;
    using resource_utility_map_t = std::unordered_map<mlir::Operation*, unsigned>;
    using schedule_time_t = size_t;

    FeasibleBarrierScheduler(mlir::MLIRContext* ctx, mlir::FuncOp func, const resource_state_t& rstate);
    FeasibleBarrierScheduler(mlir::MLIRContext* ctx, mlir::FuncOp func);

    void operator++();
    void getAllBarriersProducersAndConsumers();
    void initResourceState(const resource_state_t& start_state);
    void computeOpIndegree(operation_in_degree_t& in_degree);
    void addToCandidateSet(mlir::Operation* op);
    void computeOperationPriorities();
    void createResourceUtilityTable();
    void addOutGoingOperationsToCandidateList(mlir::Operation* op);
    void assignUniqueIds();
    void pushToHeap(const HeapElement& elem);

    bool operator==(const FeasibleBarrierScheduler& o) const;
    bool reached_end() const;
    bool nextSchedulableOperation();
    bool init(const resource_state_t& upper_bound);
    bool doesOpRunOnNCE(mlir::Operation* op);

    mlir::Operation*& operator*();
    size_t currentTime() const;
    const resource_state_t& resourceState() const;
    bool isValidOp(schedulable_ops_iterator_t itr) const;
    schedulable_ops_iterator_t find_schedulable_op();
    unsigned countProducerConsumerTasks(mlir::Operation* op);
    static SmallVector<mlir::Operation*> getConsumerOps(mlir::Operation* op);
    static std::string printOpType(VPURT::TaskOp taskOp);
    HeapElement popFromHeap();
    static mlir::IntegerAttr getUniqueID(mlir::Operation* op);

protected:
    mlir::MLIRContext* _ctx;
    mlir::FuncOp _func;
    operation_in_degree_t _in_degree;
    schedule_heap_t _heap;
    schedule_time_t _current_time;
    schedulable_ops_t _candidates;
    resource_state_t _resource_state;
    MinHeapOrdering _heap_ordering;
    mlir::Operation* _schedulable_op;
    processed_ops_t _processed_ops;
    priority_map_t _priority;
    resource_utility_map_t _resource_utility_map;
    // outputs of the graph
    llvm::DenseSet<mlir::Operation*> _outputOps;
    // operation out-degree, number of outgoing edges
    std::map<mlir::Operation*, size_t> _outDegreeTable;

    SmallVector<IERT::LayerOpInterface> _allTaskOps;
    SmallVector<VPURT::DeclareVirtualBarrierOp> _allBarrierOps;
    static std::map<mlir::Operation*, SmallVector<mlir::Operation*>> barrierProducersMap;
    static std::map<mlir::Operation*, SmallVector<mlir::Operation*>> barrierConsumersMap;
};

}  // namespace vpux
