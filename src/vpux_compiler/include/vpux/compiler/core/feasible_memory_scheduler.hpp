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
#include "vpux/compiler/utils/partitioner.hpp"

namespace vpux {

class FeasibleMemoryScheduler final {
public:
    using operationIdxType = size_t;
    // Operation type
    enum class EOpType { ORIGINAL_OP = 0, IMPLICIT_OP_READ = 1, IMPLICIT_OP_WRITE = 2, ORIGINAL_PREFETCHED_OP = 3 };
    // Operation state
    enum class EOpState { ACTIVE = 0, SPILLED = 1, CONSUMED = 2 };
    // Core struct in the feasible memory scheduler
    struct HeapElement {
        HeapElement(): op_(), time_(), opType_() {
        }
        explicit HeapElement(operationIdxType op, size_t time = 0UL, EOpType op_type = EOpType::ORIGINAL_OP,
                             mlir::Value spillBuffer = nullptr)
                : op_(op), time_(time), opType_(op_type), spillBuffer_(spillBuffer) {
        }
        bool operator==(const HeapElement& other) const {
            return (op_ == other.op_) && (time_ == other.time_) && (spillBuffer_ == other.spillBuffer_);
        }
        bool isOriginalOp() const {
            return (opType_ == EOpType::ORIGINAL_OP) || (opType_ == EOpType::ORIGINAL_PREFETCHED_OP);
        }
        bool isPrefetched() const {
            return (opType_ == EOpType::ORIGINAL_PREFETCHED_OP);
        }
        bool isImplicitWriteOp() const {
            return (opType_ == EOpType::IMPLICIT_OP_WRITE);
        }
        operationIdxType op_;
        size_t time_;
        EOpType opType_;
        mlir::Value spillBuffer_;
    };
    // Sort heap by smallest time
    struct MinHeapOrdering {
        bool operator()(const HeapElement& a, const HeapElement& b) {
            if (a.time_ == b.time_) {
                if (a.isPrefetched() && !b.isPrefetched()) {
                    return true;
                } else if (!a.isPrefetched() && b.isPrefetched()) {
                    return false;
                } else {
                    return a.op_ > b.op_;
                }
            }
            return a.time_ > b.time_;
        }
    };
    // Sort pair by the second arg
    struct SizeSort {
        bool operator()(const std::pair<operationIdxType, vpux::AddressType>& op1,
                        const std::pair<operationIdxType, vpux::AddressType>& op2) const {
            if (op1.second == op2.second) {
                return op1.first < op2.first;
            }

            return op1.second > op2.second;
        }
    };
    // Sort pair by the second arg
    struct TimeSort {
        bool operator()(const std::pair<operationIdxType, vpux::AddressType>& op1,
                        const std::pair<operationIdxType, vpux::AddressType>& op2) const {
            if (op1.second == op2.second) {
                return op1.first < op2.first;
            }

            // if time is the same sort by operationIdx which reflects IR order
            return op1.second < op2.second;
        }
    };
    // Struct used during scheduling, containing op info
    struct OpOutputInfo {
        explicit OpOutputInfo(EOpState state = EOpState::CONSUMED, size_t outstanding_consumers = 0UL)
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
        void incrementConsumers() {
            if (!outstandingConsumers_) {
                state_ = EOpState::SPILLED;
            }
            ++outstandingConsumers_;
        }
        EOpState state_;
        size_t outstandingConsumers_;
        mlir::DenseSet<size_t> spillIdx_;
    };
    // Struct storing CMX address space
    struct IntervalInfo {
        void invalidate() {
            begin_ = std::numeric_limits<size_t>::max();
            end_ = std::numeric_limits<size_t>::min();
        }
        size_t length() const {
            VPUX_THROW_UNLESS(begin_ <= end_, "Invalid resource interval");
            return end_ - begin_ + 1;
        }
        IntervalInfo(): begin_(), end_() {
            invalidate();
        }
        IntervalInfo(size_t ibeg, size_t iend): begin_(ibeg), end_(iend) {
        }
        bool operator==(const IntervalInfo& other) const {
            return (begin_ == other.begin_) && (end_ == other.end_) && (buffer_ == other.buffer_);
        }
        size_t begin_;
        size_t end_;
        mlir::Value buffer_;
    };
    // Struct used to output the scheduled op info
    struct ScheduledOpInfo {
        ScheduledOpInfo(operationIdxType op, EOpType type, size_t time, vpux::AddressType freeCmx, bool isDataOp)
                : op_(op), opType_(type), time_(time), freeCmx_(freeCmx), isDataOp_(isDataOp) {
        }
        ScheduledOpInfo(): op_(), opType_(), time_(), freeCmx_(), isDataOp_() {
        }
        bool operator==(const ScheduledOpInfo& other) const {
            return (other.op_ == op_) && (other.opType_ == opType_);
        }
        ScheduledOpInfo& operator=(const HeapElement& helement) {
            op_ = helement.op_;
            opType_ = helement.opType_;
            return *this;
        }
        bool isOriginalOp() const {
            return (opType_ == EOpType::ORIGINAL_OP) || (opType_ == EOpType::ORIGINAL_PREFETCHED_OP);
        }
        bool isSpillWrite() const {
            return (opType_ == EOpType::IMPLICIT_OP_WRITE);
        }
        bool isSpillRead() const {
            return (opType_ == EOpType::IMPLICIT_OP_READ);
        }
        bool isPrefetched() const {
            return (opType_ == EOpType::ORIGINAL_PREFETCHED_OP);
        }
        StringLiteral opTypeName() const {
            if (opType_ == EOpType::ORIGINAL_OP) {
                return StringLiteral("ORIGINAL");
            } else if (opType_ == EOpType::IMPLICIT_OP_READ) {
                return StringLiteral("SPILLED_READ");
            } else if (opType_ == EOpType::IMPLICIT_OP_WRITE) {
                return StringLiteral("SPILLED_WRITE");
            } else if (opType_ == EOpType::ORIGINAL_PREFETCHED_OP) {
                return StringLiteral("PREFETCHED");
            }
            return StringLiteral("UNDEFINED");
        }
        size_t numOfOutputResources() const {
            return outputResourceInfo_.size();
        }
        bool hasActiveOutputResource() const {
            for (size_t resourceIdx = 0; resourceIdx < numOfOutputResources(); resourceIdx++) {
                if (isActiveOutputResource(resourceIdx)) {
                    return true;
                }
            }
            return false;
        }
        bool isActiveOutputResource(size_t idx) const {
            return (outputResourceInfo_[idx].begin_ <= outputResourceInfo_[idx].end_);
        }
        bool isDataOp() const {
            return isDataOp_;
        }
        size_t resourceSize() const {
            size_t size = 0;
            for (size_t resourceIdx = 0; resourceIdx < numOfOutputResources(); resourceIdx++) {
                if (isActiveOutputResource(resourceIdx)) {
                    size += endOutputResource(resourceIdx) - beginOutputResource(resourceIdx);
                }
            }
            return size;
        }
        size_t beginOutputResource(size_t idx) const {
            return outputResourceInfo_[idx].begin_;
        }
        size_t endOutputResource(size_t idx) const {
            return outputResourceInfo_[idx].end_;
        }
        mlir::Value getOutputBuffer(size_t idx) const {
            return outputResourceInfo_[idx].buffer_;
        }

        size_t numOfInputResources() const {
            return inputResourceInfo_.size();
        }
        bool hasActiveInputResource() const {
            for (size_t resourceIdx = 0; resourceIdx < numOfInputResources(); resourceIdx++) {
                if (isActiveInputResource(resourceIdx)) {
                    return true;
                }
            }
            return false;
        }
        bool isActiveInputResource(size_t idx) const {
            return (inputResourceInfo_[idx].begin_ <= inputResourceInfo_[idx].end_);
        }
        size_t beginInputResource(size_t idx) const {
            return inputResourceInfo_[idx].begin_;
        }
        size_t endInputResource(size_t idx) const {
            return inputResourceInfo_[idx].end_;
        }
        mlir::Value getInputBuffer(size_t idx) const {
            return inputResourceInfo_[idx].buffer_;
        }

        operationIdxType op_;
        EOpType opType_;
        size_t time_;
        vpux::AddressType freeCmx_;
        bool isDataOp_;
        SmallVector<IntervalInfo> outputResourceInfo_;
        SmallVector<IntervalInfo> inputResourceInfo_;
    };
    // Struct storing eviction policy info for buffers
    struct EvictionCandidate {
        EvictionCandidate(size_t priority, size_t size, operationIdxType bufferWriterIdx, size_t outputIdx,
                          mlir::Value buffer)
                : priority_(priority),
                  size_(size),
                  bufferWriterIdx_(bufferWriterIdx),
                  outputIdx_(outputIdx),
                  buffer_(buffer) {
        }
        size_t priority_;
        size_t size_;
        operationIdxType bufferWriterIdx_;
        size_t outputIdx_;
        mlir::Value buffer_;
    };
    // Eviction priority policy for buffers
    struct EvictionPriority {
        bool operator()(const EvictionCandidate& ec1, const EvictionCandidate& ec2) const {
            // first - higher eviction priority
            if (ec1.priority_ != ec2.priority_) {
                return ec1.priority_ > ec2.priority_;
            }

            // second - smaller size
            if (ec1.size_ != ec2.size_) {
                return ec1.size_ < ec2.size_;
            }

            // third - async deps index (unique with single output)
            if (ec1.bufferWriterIdx_ != ec2.bufferWriterIdx_) {
                return ec1.bufferWriterIdx_ < ec2.bufferWriterIdx_;
            }

            // fourth - same index, multiple outputs, check idx
            return ec1.outputIdx_ > ec2.outputIdx_;
        }
    };
    using prefetchMap =
            std::unordered_map<operationIdxType, std::set<std::pair<operationIdxType, vpux::AddressType>, TimeSort>>;

public:
    FeasibleMemoryScheduler(mlir::Attribute& memSpace, MemLiveRangeInfo& liveRangeInfo, AsyncDepsInfo& depsInfo,
                            AliasesInfo& aliasInfo, Logger log, LinearScan<mlir::Value, LinearScanHandler>& scan);

public:
    SmallVector<ScheduledOpInfo> generateSchedule(prefetchMap prefetchEdges = {});

private:
    bool init();
    void clearLists();
    void nextSchedulableOp();
    void getReadyDataList();
    void getReadyComputeList();
    SmallVector<operationIdxType> reduceInDegreeOfAdjacentOperations(operationIdxType opIdx);
    bool isReadyComputeOperationSchedulable(operationIdxType opIdx);
    SmallVector<mlir::Value> getNonAliveBuffersUsedByOperation(operationIdxType opIdx);
    SmallVector<mlir::Value> sortUsedBuffers(mlir::DenseSet<mlir::Value>& operationBuffers);
    mlir::DenseSet<operationIdxType> getNonEmptyOpDemandList(operationIdxType opIdx,
                                                             llvm::ArrayRef<mlir::Value> neededBuffers);
    void scheduleInputOpForComputeOp(operationIdxType inputIdx, size_t delay);
    void scheduleSpilledOpBuffer(operationIdxType inputIdx, mlir::Value* buffer);
    size_t allocateBuffersAndInputOps(operationIdxType opIdx,
                                      Partitioner::Direction allocDir = Partitioner::Direction::Up);
    size_t scheduleComputeOp(operationIdxType opIdx);
    size_t scheduleAllPossibleReadyOpsAndUpdate();
    void schedulePrefetchedDataOps(size_t computeOpStartTime);
    void pushToStartTimeHeap(const HeapElement& elem);
    void pushToCompletionTimeHeap(const HeapElement& elem);
    HeapElement popFromStartTimeHeap();
    HeapElement popFromCompletionTimeHeap();
    HeapElement const* topElementGen(ArrayRef<HeapElement> heap) const;
    bool isDataOp(operationIdxType opIdx);
    bool isNonComputeChainOp(operationIdxType opIdx);
    bool isCopyOutOp(operationIdxType opIdx);
    void unscheduleOp(const HeapElement& helement);
    bool isComputeOpWithSomeActiveInputs(operationIdxType opIdx);
    void distributeReadyOps(llvm::ArrayRef<operationIdxType> readyOps);
    SmallVector<HeapElement> popAllElementsAtThisTime(size_t time_step);
    void unscheduleAllCompletingOpsAtNextEarliestTime();
    void populateScheduledOps(HeapElement& scheduledOp);
    vpux::AddressType calculateOpSize(operationIdxType opIdx);
    void evictActiveOp(EvictionCandidate evictionCandidate);
    size_t evictionPriority(mlir::Value buffer);
    IERT::LayerOpInterface retrieveBufferWriter(mlir::Value buffer);
    EvictionCandidate chooseCandidateForEviction(mlir::DenseSet<mlir::Value> aliveBuffers);
    void forceScheduleActiveOpEviction();
    size_t getOpBufferOutputIdx(operationIdxType opIdx, mlir::Value buffer);

private:
    Logger _log;
    // memory space, which will be allocated
    mlir::Attribute& _memSpace;
    // information about op buffers
    MemLiveRangeInfo& _liveRangeInfo;
    // dependencies of ops
    AsyncDepsInfo& _depsInfo;
    // aliases information for buffers
    AliasesInfo& _aliasInfo;
    // allocator class
    LinearScan<mlir::Value, LinearScanHandler>& _scan;
    // heap with earliest operation start time
    SmallVector<HeapElement> _startTimeHeap;
    // heap with earlies operation completion time
    SmallVector<HeapElement> _completionTimeHeap;
    // operations with ACTIVE input
    std::set<std::pair<operationIdxType, vpux::AddressType>, SizeSort> _activeComputeOps;
    // compute operations with 0 in-degree
    std::set<std::pair<operationIdxType, vpux::AddressType>, SizeSort> _readyComputeOps;
    // data operations with 0 in-degree
    std::set<std::pair<operationIdxType, vpux::AddressType>, SizeSort> _readyDataOps;
    // operations which do not belong to main compute chain for activations from network
    // input to output. Such operations need to be distinguished from other ops as scheduler
    // is focused on scheduling ops along compute chain. Such operation will only be considered
    // for scheduling once all input dependency data and/or compute ops have been executed
    std::set<std::pair<operationIdxType, vpux::AddressType>, SizeSort> _nonComputeChainOps;
    // operation in-degree, number of incoming edges
    std::unordered_map<operationIdxType, size_t> _inDegreeTable;
    // operation out-degree, number of outgoing edges
    std::unordered_map<operationIdxType, size_t> _outDegreeTable;
    // map storing prefetch edges
    prefetchMap _prefetchEdges;
    // unlocked prefetch data ops
    std::set<operationIdxType> _prefetchDataOps;
    // contains scheduled ops along with their status/type
    std::unordered_map<operationIdxType, OpOutputInfo> _opOutputTable;
    // contains the operation writing to the buffer
    std::map<mlir::Value, IERT::LayerOpInterface, vpux::ValueOrderCmp> _opWritingToBuffer;
    // container for the schedule output
    SmallVector<ScheduledOpInfo> _scheduledOps;
    // outputs of the graph
    llvm::DenseSet<operationIdxType> _outputOps;
    // schedule time
    size_t _currentTime;
};

}  // namespace vpux
