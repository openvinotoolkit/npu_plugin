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
#include "vpux/compiler/core/linear_scan_handler.hpp"
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

class FeasibleMemoryScheduler final {
public:
    // The spill op is considered an implicit op //
    enum class EOpType { ORIGINAL_OP = 0, IMPLICIT_OP_READ = 1, IMPLICIT_OP_WRITE = 2 };

    enum class EOpState { ACTIVE = 0, SPILLED = 1, CONSUMED = 2 };

    struct HeapElement {
        HeapElement(): op_(), time_(), opType_() {
        }
        HeapElement(size_t op, size_t t = 0UL, EOpType op_type = EOpType::ORIGINAL_OP)
                : op_(op), time_(t), opType_(op_type) {
        }
        bool operator==(const HeapElement& other) const {
            return (op_ == other.op_) && (time_ == other.time_);
        }
        bool isOriginalOp() const {
            return (opType_ == EOpType::ORIGINAL_OP);
        }
        bool isImplicitWriteOp() const {
            return (opType_ == EOpType::IMPLICIT_OP_WRITE);
        }
        size_t op_;
        size_t time_;
        EOpType opType_;
    };

    struct MinHeapOrdering {
        bool operator()(const HeapElement& a, const HeapElement& b) {
            return a.time_ > b.time_;
        }
    };

    struct SizeSort {
        bool operator()(const std::pair<size_t, vpux::AddressType>& op1,
                        const std::pair<size_t, vpux::AddressType>& op2) const {
            if (op1.second == op2.second) {
                return op1.first < op2.first;
            }

            return op1.second > op2.second;
        }
    };

    struct OpOutputInfo {
        OpOutputInfo(EOpState state = EOpState::CONSUMED, size_t outstanding_consumers = 0UL)
                : state_(state), outstandingConsumers_(outstanding_consumers) {
        }
        bool active() const {
            return state_ == EOpState::ACTIVE;
        }
        bool spilled() const {
            return state_ == EOpState::SPILLED;
        }
        bool consumed() const {
            return state_ == EOpState::CONSUMED;
        }
        bool hasSingleOutstandingConsumer() const {
            return outstandingConsumers_ == 1UL;
        }
        void changeStateToActive() {
            state_ = EOpState::ACTIVE;
        }
        void changeStateToConsumed() {
            state_ = EOpState::CONSUMED;
        }
        void changeStateToSpilled() {
            state_ = EOpState::SPILLED;
        }
        void decrementConsumers() {
            VPUX_THROW_UNLESS(outstandingConsumers_ > 0UL, "Invalid number of consumers");
            --outstandingConsumers_;
            if (!outstandingConsumers_) {
                state_ = EOpState::CONSUMED;
            }
        }
        EOpState state_;
        size_t outstandingConsumers_;
    };

    struct IntervalInfo {
        void invalidate() {
            begin_ = std::numeric_limits<size_t>::max();
            end_ = std::numeric_limits<size_t>::min();
        }
        size_t length() const {
            VPUX_THROW_UNLESS(begin_ <= end_, "Invalid resource interval");
            return (end_ - begin_ + 1);
        }
        IntervalInfo(): begin_(), end_() {
            invalidate();
        }
        IntervalInfo(size_t ibeg, size_t iend): begin_(ibeg), end_(iend) {
        }
        bool operator==(const IntervalInfo& other) const {
            return (begin_ == other.begin_) && (end_ == other.end_);
        }
        size_t begin_;
        size_t end_;
    };

    struct ScheduledOpInfo {
        ScheduledOpInfo(size_t op, EOpType type, size_t time): op_(op), opType_(type), time_(time) {
        }
        ScheduledOpInfo(): op_(), opType_(), time_() {
        }
        bool operator==(const ScheduledOpInfo& other) const {
            return (other.op_ == op_) && (other.opType_ == opType_);
        }
        const ScheduledOpInfo& operator=(const HeapElement& helement) {
            op_ = helement.op_;
            opType_ = helement.opType_;
            return *this;
        }
        const char* opTypeName() const {
            const char* ret = NULL;

            switch (opType_) {
            case EOpType::ORIGINAL_OP:
                ret = "ORIGINAL";
                break;
            case EOpType::IMPLICIT_OP_READ:
                ret = "SPILLED_READ";
                break;
            default:
                ret = "SPILLED_WRITE";
                break;
            }
            return ret;
        }
        bool hasActiveResource() const {
            return (resourceInfo_.begin_ <= resourceInfo_.end_);
        }
        size_t beginResource() const {
            return resourceInfo_.begin_;
        }
        size_t endResource() const {
            return resourceInfo_.end_;
        }
        size_t op_;
        EOpType opType_;
        size_t time_;
        IntervalInfo resourceInfo_;
    };

public:
    explicit FeasibleMemoryScheduler(mlir::Attribute& memSpace, MemLiveRangeInfo& liveRangeInfo,
                                     AsyncDepsInfo& depsInfo, LinearScan<mlir::Value, LinearScanHandler>& scan,
                                     mlir::Identifier timeAttrName);

public:
    llvm::SmallVector<ScheduledOpInfo> generateSchedule();

private:
    bool init();
    void clearLists();
    void nextSchedulableOp();
    void getReadyDataList();
    void getReadyComputeList();
    llvm::SmallVector<size_t> reduceInDegreeOfAdjacentOperations(size_t opIdx);
    bool isReadyComputeOperationSchedulable(size_t opIdx);
    SmallVector<mlir::Value> getSortedBuffers(size_t opIdx);
    SmallVector<size_t> getNonEmptyOpDemandList(size_t opIdx);
    void scheduleInputOpForComputeOp(size_t inputIdx);
    void allocateSortedBuffers(ArrayRef<mlir::Value> sortedBuffers);
    void scheduleComputeOp(size_t opIdx);
    void scheduleAllPossibleReadyOpsAndUpdate(std::set<std::pair<size_t, vpux::AddressType>, SizeSort>& readyList);
    void pushToStartTimeHeap(const HeapElement& elem);
    void pushToCompletionTimeHeap(const HeapElement& elem);
    HeapElement popFromStartTimeHeap();
    HeapElement popFromCompletionTimeHeap();
    HeapElement const* topElementGen(const llvm::SmallVector<HeapElement>& heap) const;
    bool isDataOp(size_t opIdx);
    void unscheduleOp(const HeapElement& helement);
    bool isComputeOpWithSomeActiveInputs(size_t opIdx);
    void distributeReadyOps(llvm::ArrayRef<size_t> readyOps);
    llvm::SmallVector<HeapElement> popAllElementsAtThisTime(size_t time_step);
    void unscheduleAllCompletingOpsAtNextEarliestTime();
    void populateScheduledOps(HeapElement& scheduledOp);
    void setTime(mlir::async::ExecuteOp execOp, size_t time);
    vpux::AddressType calculateOpSize(size_t opIdx);

private:
    mlir::Attribute& _memSpace;
    MemLiveRangeInfo& _liveRangeInfo;
    AsyncDepsInfo& _depsInfo;
    LinearScan<mlir::Value, LinearScanHandler>& _scan;
    llvm::SmallVector<HeapElement> _startTimeHeap;
    llvm::SmallVector<HeapElement> _completionTimeHeap;
    std::set<std::pair<size_t, vpux::AddressType>, SizeSort> _activeComputeOps;
    std::set<std::pair<size_t, vpux::AddressType>, SizeSort> _readyComputeOps;
    std::set<std::pair<size_t, vpux::AddressType>, SizeSort> _readyDataOps;
    std::unordered_map<size_t, size_t> _inDegreeTable;
    std::unordered_map<size_t, size_t> _outDegreeTable;
    std::unordered_map<size_t, OpOutputInfo> _opOutputTable;
    std::map<size_t, SmallVector<size_t>> _timeBuckets;  // temporary TODO: remove
    llvm::SmallVector<ScheduledOpInfo> _scheduledOps;
    llvm::DenseSet<size_t> _outputOps;
    size_t _currentTime;
    mlir::Identifier _timeAttrName;
};

}  // namespace vpux
