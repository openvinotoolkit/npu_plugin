//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/linear_scan_handler.hpp"
#include "vpux/compiler/core/mem_live_range_info.hpp"
#include "vpux/compiler/utils/partitioner.hpp"

namespace vpux {

class FeasibleMemoryScheduler final {
public:
    using operationIdxType = size_t;
    // Operation type
    enum class EOpType {
        ORIGINAL_OP = 0,
        ORIGINAL_PREFETCHED_OP = 1,
        ORIGINAL_SPILL_READ_OP = 2,
        ORIGINAL_SPILL_WRITE_OP = 3,
        IMPLICIT_SPILL_READ_OP = 4,
        IMPLICIT_SPILL_WRITE_OP = 5,
        IMPLICIT_SPILL_WRITE_FRAG_OP = 6
    };
    // Operation state
    enum class EOpState { ACTIVE = 0, SPILLED = 1, CONSUMED = 2 };
    // Core struct in the feasible memory scheduler
    struct HeapElement {
        explicit HeapElement(operationIdxType op = 0UL, llvm::BitVector executorInstanceMask = {},
                             size_t cycleBegin = 0UL, size_t cycleEnd = 0UL, EOpType op_type = EOpType::ORIGINAL_OP,
                             mlir::Value spillBuffer = nullptr)
                : op_(op),
                  executorInstanceMask_(std::move(executorInstanceMask)),
                  cycleBegin_(cycleBegin),
                  cycleEnd_(cycleEnd),
                  opType_(op_type),
                  spillBuffer_(spillBuffer) {
        }
        bool operator==(const HeapElement& other) const {
            return (op_ == other.op_) && (cycleBegin_ == other.cycleBegin_) && (spillBuffer_ == other.spillBuffer_);
        }
        bool isOriginalOp() const {
            return (opType_ == EOpType::ORIGINAL_OP) || (opType_ == EOpType::ORIGINAL_PREFETCHED_OP);
        }
        bool isPrefetched() const {
            return (opType_ == EOpType::ORIGINAL_PREFETCHED_OP);
        }
        bool isSpillWriteOp() const {
            return (opType_ == EOpType::IMPLICIT_SPILL_WRITE_OP) || (opType_ == EOpType::IMPLICIT_SPILL_WRITE_FRAG_OP);
        }
        bool isSpillReadOp() const {
            return (opType_ == EOpType::IMPLICIT_SPILL_READ_OP);
        }
        bool isSpillOp() const {
            return isSpillWriteOp() || isSpillReadOp();
        }
        operationIdxType op_;
        // Mask identyfing indexes of given executor type to be used for running this operation
        llvm::BitVector executorInstanceMask_;
        size_t cycleBegin_;
        size_t cycleEnd_;
        EOpType opType_;
        mlir::Value spillBuffer_;
    };

    // Sort heap by earliest begin cycle
    struct CycleBeginMinHeapOrdering {
        bool operator()(const HeapElement& a, const HeapElement& b);
    };

    // Sort heap by earliest end cycle
    struct CycleEndMinHeapOrdering {
        bool operator()(const HeapElement& a, const HeapElement& b);
    };

    // Sort pair by the second arg - operation index (=async-deps-index) which is aligned with order in IR
    struct operationIdxSort {
        // TODO will be replaced by DPU order heuristic
        bool operator()(const operationIdxType& op1, const operationIdxType& op2) const {
            return op1 < op2;
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
        ScheduledOpInfo(operationIdxType op, EOpType type, size_t cycleBegin, vpux::AddressType freeCmx, bool isDataOp)
                : op_(op), opType_(type), cycleBegin_(cycleBegin), freeCmx_(freeCmx), isDataOp_(isDataOp) {
        }

        ScheduledOpInfo() = default;

        bool operator==(const ScheduledOpInfo& other) const {
            return (other.op_ == op_) && (other.opType_ == opType_);
        }
        ScheduledOpInfo& operator=(const HeapElement& helement) {
            op_ = helement.op_;
            opType_ = helement.opType_;
            return *this;
        }
        bool isOriginalOp() const {
            return (opType_ == EOpType::ORIGINAL_OP) || (opType_ == EOpType::ORIGINAL_PREFETCHED_OP) ||
                   (opType_ == EOpType::ORIGINAL_SPILL_WRITE_OP) || (opType_ == EOpType::ORIGINAL_SPILL_READ_OP);
        }
        bool isOriginalSpillWriteOp() const {
            return (opType_ == EOpType::ORIGINAL_SPILL_WRITE_OP);
        }
        bool isOriginalSpillReadOp() const {
            return (opType_ == EOpType::ORIGINAL_SPILL_READ_OP);
        }
        bool isSpillWrite() const {
            return (opType_ == EOpType::IMPLICIT_SPILL_WRITE_OP) || (opType_ == EOpType::IMPLICIT_SPILL_WRITE_FRAG_OP);
        }
        bool isSpillWriteFrag() const {
            return (opType_ == EOpType::IMPLICIT_SPILL_WRITE_FRAG_OP);
        }
        bool isSpillRead() const {
            return (opType_ == EOpType::IMPLICIT_SPILL_READ_OP);
        }
        bool isPrefetched() const {
            return (opType_ == EOpType::ORIGINAL_PREFETCHED_OP);
        }
        StringLiteral opTypeName() const {
            if (opType_ == EOpType::ORIGINAL_OP) {
                return StringLiteral("ORIGINAL");
            } else if (opType_ == EOpType::ORIGINAL_PREFETCHED_OP) {
                return StringLiteral("ORIGINAL_PREFETCHED");
            } else if (opType_ == EOpType::ORIGINAL_SPILL_READ_OP) {
                return StringLiteral("ORIGINAL_SPILL_READ");
            } else if (opType_ == EOpType::ORIGINAL_SPILL_WRITE_OP) {
                return StringLiteral("ORIGINAL_SPILL_WRITE");
            } else if (opType_ == EOpType::IMPLICIT_SPILL_READ_OP) {
                return StringLiteral("IMPLICIT_SPILL_READ");
            } else if (opType_ == EOpType::IMPLICIT_SPILL_WRITE_OP) {
                return StringLiteral("IMPLICIT_SPILL_WRITE");
            } else if (opType_ == EOpType::IMPLICIT_SPILL_WRITE_FRAG_OP) {
                return StringLiteral("IMPLICIT_SPILL_WRITE_FRAG");
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

        operationIdxType op_{};
        EOpType opType_{EOpType::ORIGINAL_OP};
        size_t cycleBegin_{};
        vpux::AddressType freeCmx_{};
        bool isDataOp_{false};
        SmallVector<IntervalInfo> outputResourceInfo_;
        SmallVector<IntervalInfo> inputResourceInfo_;
        size_t cycleEnd_{};
        VPU::ExecutorKind executor{VPU::ExecutorKind::DMA_NN};
        llvm::BitVector executorInstanceMask;
        bool isNonComputeChain{false};
    };
    using ScheduledOpInfoVec = SmallVector<ScheduledOpInfo, 1>;

    // Struct storing buffer info for allocation order
    struct BufferOrder {
        BufferOrder(mlir::Value buffer, vpux::AddressType size, size_t outDegree,
                    size_t level = std::numeric_limits<size_t>::min(), bool highAllocationPriority = false)
                : buffer(buffer),
                  size(size),
                  outDegree(outDegree),
                  level(level),
                  highAllocationPriority(highAllocationPriority) {
        }
        mlir::Value buffer;
        vpux::AddressType size;
        size_t outDegree;
        size_t level;
        bool highAllocationPriority;
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
    // Struct storing info for prefetch DMAs
    struct PrefetchDMA {
        PrefetchDMA(operationIdxType opIdx, size_t level, size_t outDegree = 0, mlir::Value buffer = nullptr)
                : opIdx_(opIdx), level_(level), buffer_(buffer), outDegree_(outDegree) {
        }
        operationIdxType opIdx_;
        size_t level_;
        mlir::Value buffer_;
        size_t outDegree_;
    };
    // Sort prefetchSet based on PrefetchDMA level
    struct LevelSort {
        bool operator()(const PrefetchDMA& op1, const PrefetchDMA& op2) const {
            if (op1.level_ == op2.level_) {
                if (op1.outDegree_ != op2.outDegree_) {
                    return op1.outDegree_ > op2.outDegree_;
                }
                return op1.opIdx_ < op2.opIdx_;
            }

            // if level is equal sort based on IR order
            return op1.level_ < op2.level_;
        }
    };
    // Store DMA idx along with DMA level
    using prefetchSet = std::set<PrefetchDMA, LevelSort>;
    // Store DPU with DMAs which should ideally execute during this DPU
    struct OverlappedSchedule {
        OverlappedSchedule(operationIdxType computeOpIdx, size_t computeOpLevel, prefetchSet dataOps,
                           bool activationSpill = false)
                : computeOpIdx(computeOpIdx),
                  computeOpLevel(computeOpLevel),
                  dataOps(std::move(dataOps)),
                  activationSpill(activationSpill) {
        }

        operationIdxType computeOpIdx;
        size_t computeOpLevel;
        prefetchSet dataOps;
        bool activationSpill;
        // DMAs which also can be prefetched
        SmallVector<EvictionCandidate> spilledOps;
    };

    using scheduleWithPrefetch = std::list<OverlappedSchedule>;

    struct ExecutorAndCycleType {
        VPU::ExecutorKind execType;
        llvm::BitVector execMask;
        size_t cycle;
    };

public:
    FeasibleMemoryScheduler(VPU::MemoryKind memSpace, MemLiveRangeInfo& liveRangeInfo, AsyncDepsInfo& depsInfo,
                            AliasesInfo& aliasInfo, Logger log, LinearScan<mlir::Value, LinearScanHandler>& scan,
                            VPU::ArchKind arch, std::shared_ptr<VPUNN::VPUCostModel> costModel, int64_t nceClusterCount,
                            int64_t dmaCount, bool enableScheduleStatistics);

public:
    ScheduledOpInfoVec generateSchedule(const scheduleWithPrefetch& prefetchSchedule = {});

private:
    bool init();
    void clearLists();
    void schedulingLoop();
    void initializeReadyLists();
    SmallVector<operationIdxType> reduceInDegreeOfAdjacentOperations(operationIdxType opIdx);
    bool isReadyComputeOperationSchedulable(operationIdxType opIdx);
    bool hasBuffersInTargetMemoryKind(operationIdxType opIdx);
    SmallVector<mlir::Value> getNonAliveBuffersUsedByOperation(operationIdxType opIdx);
    SmallVector<mlir::Value> sortUsedBuffers(mlir::DenseSet<mlir::Value>& operationBuffers);
    mlir::DenseSet<operationIdxType> getNonEmptyOpDemandList(operationIdxType opIdx,
                                                             llvm::ArrayRef<mlir::Value> neededBuffers);
    size_t scheduleInputOpForComputeOp(operationIdxType inputIdx);
    size_t schedulePrefetchOp(operationIdxType inputIdx);
    size_t scheduleSpilledOpBuffer(operationIdxType inputIdx, mlir::Value* buffer);
    size_t getOperationEndCycle(operationIdxType opIdx, size_t nextScheduleCycle);
    size_t getEarliestComputeBeginCycle(operationIdxType opIdx);
    size_t allocateBuffersAndInputOps(operationIdxType opIdx,
                                      Partitioner::Direction allocDir = Partitioner::Direction::Up);
    void allocatePrefetchOps(operationIdxType opIdx, size_t earliestComputeBeginCycle,
                             mlir::DenseSet<mlir::Value>& buffersNeedingAllocation);
    size_t scheduleComputeOp(operationIdxType opIdx);
    void scheduleAllPossibleReadyOpsAndUpdate();
    void pushToCycleBeginHeap(const HeapElement& elem);
    void pushToCycleEndHeap(const HeapElement& elem);
    HeapElement popFromCycleBeginHeap();
    HeapElement popFromCycleEndHeap();
    HeapElement const* topElementGen(ArrayRef<HeapElement> heap) const;
    VPU::ExecutorKind getExecutorType(operationIdxType opIdx);
    size_t getCurrentCycle(operationIdxType opIdx, bool spilled = false);
    // Based on operation/buffer properties determine how many executor instances are needed
    // to run given operation
    size_t getOpDemandForExecutorsInstances(operationIdxType opIdx, VPU::ExecutorKind executorType);
    size_t getBufferDemandForExecutorsInstances(mlir::Value buffer, VPU::ExecutorKind executorType);
    // Return mask of executor instances specyfing on which executor indexes given operation
    // is to be executed on
    llvm::BitVector getExecutorInstanceMask(size_t numOfNeededInstances, VPU::ExecutorKind executorType);
    llvm::BitVector getExecutorInstanceMaskForOp(operationIdxType opIdx, VPU::ExecutorKind executorType);
    llvm::BitVector getExecutorInstanceMaskForBuffer(mlir::Value buffer, VPU::ExecutorKind executorType);
    ExecutorAndCycleType getCurrentCycleAndExecutorInstanceMask(operationIdxType opIdx);
    ExecutorAndCycleType getCurrentCycleAndExecutorInstanceMaskForSpill(mlir::Value buffer);
    size_t spilledOperationCycleCost(mlir::Value spilledBuffer);
    size_t operationCycleCost(operationIdxType opIdx);
    void updateCurrentCycleForExecutor(VPU::ExecutorKind executor, llvm::BitVector executorInstanceMask,
                                       size_t newCycle);
    void alignExecutors(size_t nextAvailableCycle);
    bool isDataOp(operationIdxType opIdx);
    bool isNonComputeChainOp(operationIdxType opIdx);
    bool isCopyOutOp(operationIdxType opIdx);
    void unscheduleOp(const HeapElement& helement);
    void distributeReadyOps(llvm::ArrayRef<operationIdxType> readyOps);
    void unscheduleAllCompletingOps();
    void populateScheduledOps(HeapElement& scheduledOp);
    void evictActiveOp(EvictionCandidate evictionCandidate);
    size_t evictionPriority(operationIdxType writerOpIdx, mlir::Value buffer);
    VPUIP::LayerOpInterface retrieveBufferWriter(mlir::Value buffer);
    EvictionCandidate chooseCandidateForEviction(const mlir::DenseSet<mlir::Value>& aliveBuffers);
    void forceScheduleActiveOpEviction();
    size_t getOpBufferOutputIdx(operationIdxType opIdx, mlir::Value buffer);
    void cleanUpAndLogSchedule();
    void createBufferAsyncIdxMap();

private:
    Logger _log;
    // memory space, which will be allocated
    VPU::MemoryKind _memKind;
    // information about op buffers
    MemLiveRangeInfo& _liveRangeInfo;
    // dependencies of ops
    AsyncDepsInfo& _depsInfo;
    // aliases information for buffers
    AliasesInfo& _aliasInfo;
    // allocator class
    LinearScan<mlir::Value, LinearScanHandler>& _scan;
    // architecture kind
    VPU::ArchKind _archKind;
    // VPUNN cost model
    std::shared_ptr<VPUNN::VPUCostModel> _costModel;
    // NCE cluster count
    int64_t _nceClusterCount;
    // Flag for enabling additional statistic related logic
    bool _enableScheduleStatistics;
    // there are 8 barriers per cluster
    int64_t _barrierPerCluster = 8;
    // heap with earliest operation start cycle
    SmallVector<HeapElement> _cycleBeginHeap;
    // heap with earliest operation end cycle
    SmallVector<HeapElement> _cycleEndHeap;
    // compute operations with 0 in-degree
    std::set<operationIdxType, operationIdxSort> _readyComputeOps;
    // data operations with 0 in-degree
    std::set<operationIdxType, operationIdxSort> _readyDataOps;
    // operations which do not belong to main compute chain for activations from network
    // input to output. Such operations need to be distinguished from other ops as scheduler
    // is focused on scheduling ops along compute chain. Such operation will only be considered
    // for scheduling once all input dependency data and/or compute ops have been executed
    std::set<operationIdxType, operationIdxSort> _nonComputeChainOps;
    // operation in-degree, number of incoming edges
    std::unordered_map<operationIdxType, size_t> _inDegreeTable;
    // operation out-degree, number of outgoing edges
    std::unordered_map<operationIdxType, size_t> _outDegreeTable;
    // map storing prefetch edges
    scheduleWithPrefetch _prefetchSchedule;
    // level of DMAs corresponding to how many DPUs will execute before this DMA is needed
    mlir::DenseMap<operationIdxType, size_t> _dataOpLevels;
    // level of buffers corresponding to how many DPUs will execute before this buffer is needed
    mlir::DenseMap<mlir::Value, size_t> _bufferLevels;
    // contains scheduled ops along with their status/type
    std::unordered_map<operationIdxType, OpOutputInfo> _opOutputTable;
    // contains the operation writing to the buffer
    mlir::DenseMap<mlir::Value, operationIdxType> _opIdxWritingToBuffer;
    // container for the schedule output
    ScheduledOpInfoVec _scheduledOps;
    // outputs of the graph
    llvm::DenseSet<operationIdxType> _outputOps;
    // cycle pipeline for every executor. Vector element type is to support multiple instances of
    // same executor type in case where scheduler needs to be aware of it
    // TODO: Currently scheduler supports only multiple DMA executors (ports)
    llvm::DenseMap<VPU::ExecutorKind, SmallVector<size_t>> _executorPipelines = {
            {VPU::ExecutorKind::DMA_NN, {1}}, {VPU::ExecutorKind::DPU, {1}},      {VPU::ExecutorKind::SHAVE_UPA, {1}},
            {VPU::ExecutorKind::NCE, {1}},    {VPU::ExecutorKind::SHAVE_NN, {1}}, {VPU::ExecutorKind::SHAVE_ACT, {1}},
    };
    // spilled operation cycle cost
    mlir::DenseMap<mlir::Value, size_t> _spillBufferCycleCost;

    // Map of root buffers and vector of operations (operationIdx) writing to this buffer
    // This is used to sort the shared buffers based on their outdegree using the _outDegreeTable
    // so that we can prefetch them at the start in the CMX not somewhere in the middle to get a contigious CMX
    // space
    mlir::DenseMap<mlir::Value, SmallVector<operationIdxType>> _bufferOpIdxMap;
};

}  // namespace vpux
