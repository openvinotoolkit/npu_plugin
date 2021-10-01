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

#include "vpux/compiler/core/async_deps_info.hpp"
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

namespace vpux {

//
// LinearScanHandler
//

class LinearScanHandler final {
public:
    explicit LinearScanHandler(AddressType defaultAlignment = 1): _defaultAlignment(defaultAlignment) {
    }

public:
    void markAsDead(mlir::Value val) {
        _aliveValues.erase(val);
    }
    void markAsAlive(mlir::Value val) {
        _aliveValues.insert(val);
    }

    Byte maxAllocatedSize() const {
        return _maxAllocatedSize;
    }

public:
    bool isAlive(mlir::Value val) const {
        return _aliveValues.contains(val);
    }

    static bool isFixedAlloc(mlir::Value val) {
        return val.getDefiningOp<IERT::StaticAllocOp>() != nullptr;
    }

    static AddressType getSize(mlir::Value val) {
        const auto type = val.getType().dyn_cast<mlir::MemRefType>();
        VPUX_THROW_UNLESS(type != nullptr, "StaticAllocation can work only with MemRef Type, got '{0}'", val.getType());

        const Byte totalSize = getTypeTotalSize(type);
        return checked_cast<AddressType>(totalSize.count());
    }

    AddressType getAlignment(mlir::Value val) const {
        if (auto allocOp = val.getDefiningOp<mlir::memref::AllocOp>()) {
            if (auto alignment = allocOp.alignment()) {
                return checked_cast<AddressType>(alignment.getValue());
            }
        }

        return _defaultAlignment;
    }

    AddressType getAddress(mlir::Value val) const {
        if (auto staticAllocOp = val.getDefiningOp<IERT::StaticAllocOp>()) {
            return checked_cast<AddressType>(staticAllocOp.offset());
        }

        const auto it = _valOffsets.find(val);
        VPUX_THROW_UNLESS(it != _valOffsets.end(), "Value '{0}' was not allocated", val);

        return it->second;
    }

    void allocated(mlir::Value val, AddressType addr) {
        VPUX_THROW_UNLESS(addr != InvalidAddress, "Trying to assign invalid address");
        VPUX_THROW_UNLESS(_valOffsets.count(val) == 0, "Value '{0}' was already allocated", val);

        _valOffsets.insert({val, addr});

        const auto endAddr = alignVal<int64_t>(addr + getSize(val), getAlignment(val));
        _maxAllocatedSize = Byte(std::max(_maxAllocatedSize.count(), endAddr));
    }

    std::list<std::pair<mlir::Value, AddressType>> getSortedAlive() {
        std::list<std::pair<mlir::Value, AddressType>> aliveWithSize;
        for (auto alive : _aliveValues) {
            AddressType size = getSize(alive);
            aliveWithSize.push_back(std::make_pair(alive, size));
            alive.dump();
        }
        // std::sort(aliveWithSize.begin(), aliveWithSize.end(),
        //           [](const std::pair<mlir::Value, AddressType>& val1, const std::pair<mlir::Value, AddressType>&
        //           val2) {
        //               return val1.second < val2.second;
        //           });
        return aliveWithSize;
    }

    void freed(mlir::Value val) {
        markAsDead(val);
    }

    static int getSpillWeight(mlir::Value) {
        VPUX_THROW("Spills are not implemented");
    }

    static bool spilled(mlir::Value) {
        VPUX_THROW("Spills are not implemented");
    }

private:
    mlir::DenseMap<mlir::Value, AddressType> _valOffsets;
    mlir::DenseSet<mlir::Value> _aliveValues;
    AddressType _defaultAlignment = 1;
    Byte _maxAllocatedSize;
};

class ListScheduler final {
    // The spill op is considered an implicit op //
    enum class op_type_e { ORIGINAL_OP = 0, IMPLICIT_OP_READ = 1, IMPLICIT_OP_WRITE = 2 };

    enum class operation_output_e { ACTIVE = 0, SPILLED = 1, CONSUMED = 2 };

    struct heap_element_t {
        heap_element_t(): op_(), time_(), op_type_() {
        }
        heap_element_t(size_t op, size_t t = 0UL, op_type_e op_type = op_type_e::ORIGINAL_OP)
                : op_(op), time_(t), op_type_(op_type) {
        }
        bool operator==(const heap_element_t& o) const {
            return (op_ == o.op_) && (time_ == o.time_);
        }
        bool is_original_op() const {
            return (op_type_ == op_type_e::ORIGINAL_OP);
        }
        bool is_implicit_write_op() const {
            return (op_type_ == op_type_e::IMPLICIT_OP_WRITE);
        }
        size_t op_;
        size_t time_;
        op_type_e op_type_;
    };

    struct min_heap_ordering_t {
        bool operator()(const heap_element_t& a, const heap_element_t& b) {
            return a.time_ > b.time_;
        }
    };

    struct op_output_info_t {
        op_output_info_t(operation_output_e state = operation_output_e::CONSUMED, size_t outstanding_consumers = 0UL)
                : state_(state), outstanding_consumers_(outstanding_consumers) {
        }
        bool active() const {
            return state_ == operation_output_e::ACTIVE;
        }
        bool spilled() const {
            return state_ == operation_output_e::SPILLED;
        }
        bool consumed() const {
            return state_ == operation_output_e::CONSUMED;
        }
        bool has_single_outstanding_consumer() const {
            return outstanding_consumers_ == 1UL;
        }
        void change_state_to_active() {
            state_ = operation_output_e::ACTIVE;
        }
        void change_state_to_consumed() {
            state_ = operation_output_e::CONSUMED;
        }
        void change_state_to_spilled() {
            state_ = operation_output_e::SPILLED;
        }
        void decrement_consumers() {
            assert(outstanding_consumers_ > 0UL);
            --outstanding_consumers_;
            if (!outstanding_consumers_) {
                state_ = operation_output_e::CONSUMED;
            }
        }
        operation_output_e state_;
        size_t outstanding_consumers_;
    };

    struct interval_info_t {
        void invalidate() {
            begin_ = std::numeric_limits<size_t>::max();
            end_ = std::numeric_limits<size_t>::min();
        }
        size_t length() const {
            assert(begin_ <= end_);
            return (end_ - begin_ + 1);
        }
        interval_info_t(): begin_(), end_() {
            invalidate();
        }
        interval_info_t(size_t ibeg, size_t iend): begin_(ibeg), end_(iend) {
        }
        bool operator==(const interval_info_t& o) const {
            return (begin_ == o.begin_) && (end_ == o.end_);
        }
        size_t begin_;
        size_t end_;
    };

    struct scheduled_op_info_t {
        scheduled_op_info_t(size_t op, op_type_e type, size_t time): op_(op), op_type_(type), time_(time) {
        }
        scheduled_op_info_t(): op_(), op_type_(), time_() {
        }
        bool operator==(const scheduled_op_info_t& o) const {
            return (o.op_ == op_) && (o.op_type_ == op_type_);
        }
        const scheduled_op_info_t& operator=(const heap_element_t& helement) {
            op_ = helement.op_;
            op_type_ = helement.op_type_;
            return *this;
        }
        const char* op_type_name() const {
            const char* ret = NULL;

            switch (op_type_) {
            case op_type_e::ORIGINAL_OP:
                ret = "ORIGINAL";
                break;
            case op_type_e::IMPLICIT_OP_READ:
                ret = "SPILLED_READ";
                break;
            default:
                ret = "SPILLED_WRITE";
                break;
            }
            return ret;
        }
        bool has_active_resource() const {
            return (resource_info_.begin_ <= resource_info_.end_);
        }
        size_t begin_resource() const {
            return resource_info_.begin_;
        }
        size_t end_resource() const {
            return resource_info_.end_;
        }
        size_t op_;
        op_type_e op_type_;
        size_t time_;
        interval_info_t resource_info_;
    };

public:
    explicit ListScheduler(mlir::Attribute& memSpace, MemLiveRangeInfo& liveRangeInfo, AsyncDepsInfo& depsInfo,
                           LinearScan<mlir::Value, LinearScanHandler>& scan, mlir::Identifier timeAttrName);

public:
    void generateSchedule();
    void addDependencies();

private:
    bool init();
    void clear_lists();
    void next_schedulable_op();
    void compute_ready_data_list();
    void compute_ready_compute_list();
    std::list<size_t> reduce_in_degree_of_adjacent_operations(size_t opIdx);
    bool is_ready_compute_operation_schedulable(size_t opIdx);
    SmallVector<mlir::Value> get_sorted_buffers(size_t opIdx);
    SmallVector<size_t> get_non_empty_op_demand_list(size_t opIdx);
    void schedule_input_op_for_compute_op(size_t inputIdx);
    void allocate_sorted_buffers(SmallVector<mlir::Value> sortedBuffers);
    void schedule_compute_op(size_t opIdx);
    void schedule_all_possible_ready_ops_and_update(std::unordered_set<size_t>& readyList);
    void push_to_st_heap(const heap_element_t& elem);
    void push_to_ct_heap(const heap_element_t& elem);
    heap_element_t pop_from_st_heap();
    heap_element_t pop_from_ct_heap();
    heap_element_t const* top_element_gen(const std::vector<heap_element_t>& heap) const;
    bool is_data_op(size_t opIdx);
    void unschedule_op(const heap_element_t& helement);
    bool is_compute_op_with_some_active_inputs(size_t opIdx);
    void distribute_ready_ops(std::list<size_t> readyOps);
    std::vector<heap_element_t> pop_all_elements_at_this_time(size_t time_step);
    void unschedule_all_completing_ops_at_next_earliest_time();
    void evict_active_op(size_t opIdx);
    void force_schedule_active_op_eviction();
    void setTime(mlir::async::ExecuteOp execOp, size_t time);

private:
    mlir::Attribute& _memSpace;
    MemLiveRangeInfo& _liveRangeInfo;
    AsyncDepsInfo& _depsInfo;
    LinearScan<mlir::Value, LinearScanHandler>& _scan;
    std::vector<heap_element_t> _startTimeHeap;
    std::vector<heap_element_t> _completionTimeHeap;
    std::unordered_set<size_t> _activeComputeOps;
    std::unordered_set<size_t> _readyComputeOps;
    std::unordered_set<size_t> _readyDataOps;
    std::unordered_map<size_t, size_t> _inDegreeTable;
    std::unordered_map<size_t, size_t> _outDegreeTable;
    std::map<size_t, SmallVector<size_t>> _timeBuckets;  // temporary TODO: remove
    std::list<scheduled_op_info_t> _scheduledOps;
    std::unordered_map<size_t, op_output_info_t> _opOutputTable;
    size_t _currentTime;
    mlir::Identifier _timeAttrName;
};

}  // namespace vpux
