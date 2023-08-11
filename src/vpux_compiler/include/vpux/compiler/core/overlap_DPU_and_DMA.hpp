//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler.hpp"

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include <llvm/ADT/BitVector.h>

namespace vpux {

class OverlapDMAandDPU final {
    using ScheduledOpInfo = vpux::FeasibleMemoryScheduler::ScheduledOpInfo;
    using scheduledOps = vpux::FeasibleMemoryScheduler::ScheduledOpInfoVec;
    using operationIdxType = vpux::FeasibleMemoryScheduler::operationIdxType;
    using OverlappedSchedule = vpux::FeasibleMemoryScheduler::OverlappedSchedule;
    using scheduleWithPrefetch = vpux::FeasibleMemoryScheduler::scheduleWithPrefetch;
    using PrefetchDMA = vpux::FeasibleMemoryScheduler::PrefetchDMA;
    using prefetchSet = vpux::FeasibleMemoryScheduler::prefetchSet;

    // stores a cycle interval for an operation
    struct CycleInterval {
        explicit CycleInterval(operationIdxType opIdx, size_t cycleBegin, size_t cycleEnd)
                : opIdx_(opIdx), cycleBegin_(cycleBegin), cycleEnd_(cycleEnd) {
        }
        size_t intervalSize() {
            return cycleEnd_ - cycleBegin_;
        }
        size_t opIdx_;
        size_t cycleBegin_;
        size_t cycleEnd_;
    };

    // stores optimal cycles for a DMA so that it completes before the consuming DPU, updated dynamically
    struct DataOpOptimalCycles {
        explicit DataOpOptimalCycles(size_t opIdx, size_t DPUIdx, size_t level, size_t cycleBegin, size_t cycleEnd)
                : opIdx_(opIdx), DPUIdx_(DPUIdx), level_(level), cycleBegin_(cycleBegin), cycleEnd_(cycleEnd) {
        }
        bool operator==(const DataOpOptimalCycles& other) const {
            return opIdx_ == other.opIdx_;
        }
        size_t cycleCost() {
            return cycleEnd_ - cycleBegin_;
        }
        // DMA index
        size_t opIdx_;
        // DPU index to which the DMA belongs
        size_t DPUIdx_;
        // level of the DMA
        size_t level_;
        // cycle when DMA will start
        size_t cycleBegin_;
        // cycle when DMA will end
        size_t cycleEnd_;
    };

    // stores optimal cycles for a DPU so that it starts right after dependencies finish
    struct DPUOptimalCycles {
        explicit DPUOptimalCycles(operationIdxType opIdx, size_t level, size_t cycleBegin, size_t cycleEnd)
                : opIdx_(opIdx),
                  level_(level),
                  cycleBegin_(cycleBegin),
                  cycleEnd_(cycleEnd),
                  cycleCost_(cycleEnd - cycleBegin) {
        }
        // DPU index
        operationIdxType opIdx_;
        // level of DPU
        size_t level_;
        size_t cycleBegin_;
        size_t cycleEnd_;
        size_t cycleCost_;
    };

    // stores activation copy outs writes and reads
    struct ActivationSpillCost {
        ActivationSpillCost() = default;

        explicit ActivationSpillCost(size_t level): level_(level) {
            reads_.clear();
            writes_.clear();
        }
        bool hasActivationSpill() {
            return activationSpillCycleCost() > 0;
        }
        size_t activationSpillCycleCost() {
            size_t totalCost = 0;
            for (auto& read : reads_) {
                totalCost += read.intervalSize();
            }
            for (auto& write : writes_) {
                totalCost += write.intervalSize();
            }
            return totalCost;
        }
        void addActivationSpillRead(operationIdxType opIdx, size_t cycleBegin, size_t cycleEnd) {
            reads_.push_back(CycleInterval(opIdx, cycleBegin, cycleEnd));
        }
        void addActivationSpillWrite(operationIdxType opIdx, size_t cycleBegin, size_t cycleEnd) {
            writes_.push_back(CycleInterval(opIdx, cycleBegin, cycleEnd));
        }
        // level at which activation copy out is occuring
        size_t level_{};
        // activation read DMAs
        SmallVector<CycleInterval> reads_;
        // activation write DMAs
        SmallVector<CycleInterval> writes_;
    };

    // free DMA cycles overlapping with DPU
    struct FreeInterval {
        explicit FreeInterval(size_t cycleBegin, size_t cycleEnd);
        // util to print the pipeline
        void printFreeInterval();
        // reduce cycles and move read to free interval
        void overlapActivationCopyIn(CycleInterval activationCopyInCycles, size_t copyInLevel,
                                     bool thisIntervalActivationSpill);
        // calculate size of interval
        size_t intervalSize(size_t cycleEnd);
        // add DMA to interval
        void addOpToFreeInterval(DataOpOptimalCycles dataOp);
        // add DMA to interval and stall pipeline
        void addOpsToFreeIntervalWithActivationStall(SmallVector<DataOpOptimalCycles> dataOps);
        // stall interval based on a fixed stall
        void addFreeIntervalStall(size_t stall);
        // stall interval based on cycles
        void addFreeIntervalStall(size_t oldCycle, size_t newCycle);
        // find DMAs of required cycle size
        SmallVector<DataOpOptimalCycles> frontCandidatesOfSize(size_t size);
        // check if interval has free cycles for DMA to complete optimally
        bool canOpFitInInterval(DataOpOptimalCycles dataOp);
        // check if cycles required are free
        bool canOpFitInInterval(size_t cycles);
        // return free cycles in this interval
        size_t getIntervalFreeCycles();

        size_t cycleBegin_;
        size_t cycleEnd_;
        // operations added to execute during this free interval
        SmallVector<DataOpOptimalCycles> prefetchOps_;
    };

    // cycles during which activation copy out occurs
    struct ActivationSpill {
        explicit ActivationSpill(size_t cycleBegin = std::numeric_limits<size_t>::max(),
                                 size_t cycleEnd = std::numeric_limits<size_t>::min(),
                                 size_t level = std::numeric_limits<size_t>::max());
        // check if valid activation copy out
        bool hasActivationSpill();
        // activation copy out total cost in cycles
        size_t activationSpillCycleCost();
        // util to print the pipeline
        void printActivationSpill();
        // stall interval based on a fixed stall
        void addActivationSpillStall(size_t stall);
        // check if current interval has activation read index
        bool hasReadWithIdx(operationIdxType idx);
        // reduce cycles and move read to free interval
        void overlapActivationCopyIn(CycleInterval activationCopyInCycles, bool thisIntervalActivationSpill);
        // add activation read and write cycles
        void populateReadsAndWrites(ArrayRef<CycleInterval> reads, ArrayRef<CycleInterval> writes);
        // stall interval based on cycles
        void addActivationSpillStall(size_t oldCycle, size_t newCycle);

        size_t cycleBegin_;
        size_t cycleEnd_;
        size_t level_;
        std::vector<CycleInterval> reads_;
        std::vector<CycleInterval> writes_;
    };

    // execution of operations until an activation copy out occurs with overlap during free interval
    struct PrefetchInterval {
        explicit PrefetchInterval(FreeInterval freeInterval, SmallVector<DPUOptimalCycles> DPUOptimalCycles,
                                  ActivationSpill activationSpill);
        // util to print the pipeline
        void printPrefetchInterval();
        // reduce DPU pipeline by overlapped activation copy in
        void overlapActivationCopyInForDPU(CycleInterval activationCopyInCycles, bool thisIntervalActivationSpill);
        // reduce interval by overlapped activation copy in
        void overlapActivationCopyIn(CycleInterval activationCopyInCycles, bool thisIntervalActivationSpill,
                                     size_t level);
        // check if interval has free cycles for DMA to complete optimally
        bool canOpFitInInterval(DataOpOptimalCycles dataOp);
        // check if cycles required are free
        bool canOpFitInInterval(size_t cycles);
        // add DMA to interval during optimal or earlier cycles
        void addOpToInterval(DataOpOptimalCycles dataOp);
        // update DPU cycles based on input DMAs
        size_t updateDPUCycles();
        // add DMAs which delay DPU but enable overlap
        size_t addOpsToIntervalWithActivationStall(SmallVector<DataOpOptimalCycles> dataOps);
        // update DPU cycle look up based on input DMAs
        void updateDPULookUp(ArrayRef<DPUOptimalCycles> DPUOptimalCycles);
        // stall DPUs based on a fixed stall
        void addDPUStall(size_t stall);
        // stall DPUs based on cycles
        void addDPUStall(size_t oldCycle, size_t newCycle);
        // stall interval based on a fixed stall
        void stallIntervals(size_t stall);
        // stall interval based on cycles
        void stallIntervals(size_t oldCycle, size_t newCycle);
        // return candidates to free required cycles
        SmallVector<DataOpOptimalCycles> getCandidatesGreaterThan(size_t size);
        // total size of free cycles
        size_t freeIntervalTotalSize(size_t cycleEnd);
        // interval cycle start
        size_t getIntervalStart();
        // interval cycle end
        size_t getIntervalEnd();
        // return DMAs executing during free interval
        ArrayRef<DataOpOptimalCycles> getDataOpCyclesDuringFreeInterval();
        // reutn DPUs executing during this interval
        ArrayRef<DPUOptimalCycles> getDPUCyclesDuringPrefetchInterval();
        // check if DPU is executing during this interval
        bool isDPUExecutingDuringInterval(operationIdxType opIdx);
        // get optimal DPU start cycle
        size_t getDPUBeginCycle(operationIdxType opIdx);

        FreeInterval freeInterval_;
        SmallVector<DPUOptimalCycles> DPUOptimalCycles_;
        mlir::DenseMap<operationIdxType, size_t> optimalDPUStartTime_;
        ActivationSpill activationSpill_;
    };

    // execution of operations with overlap
    struct PrefetchPipeline {
        explicit PrefetchPipeline();
        // add a new interval consisting of free interval, DPU optimal cycles, and activation copy out
        void addPrefetchInterval(PrefetchInterval interval);
        // return all activation copy outs from all prefetch intervals
        SmallVector<ActivationSpill> getActivationSpills();
        // try to prefetch activation copy in DMA
        void overlapActivationCopyIn(CycleInterval activationCopyInCycles, size_t level);
        // util to print the pipeline
        void printPrefetchPipeline();
        // delay all prefetch intervals after old cycle with new cycle
        void stallPipeline(size_t oldCycle, size_t newCycle);
        // calculate optimal DPU start time on DPU pipeline
        size_t getOptimalDPUBeginCycle(operationIdxType opIdx);
        // set the DMA to complete before the optimal DPU cycle start
        void updateDMAOptimalCycles(DataOpOptimalCycles* dataOp);
        // return all DPU optimal cycles from all prefetch intervals
        SmallVector<DPUOptimalCycles> getOptimalDPUCycles();
        // link DMAs to DPUs based on intervals and cycles
        void populatePrefetchEdges(scheduleWithPrefetch& prefetchSchedule);
        // find at which interval a DMA would execute optimally
        PrefetchInterval* findTargetInterval(size_t cycleEnd);
        // move DMAs to previous intervals to make cycle space
        bool shiftToPreviousInterval(PrefetchInterval* currentInterval, DataOpOptimalCycles dataOp);
        // data bound case where DPU might need to be delayed in order to prefetch
        void prefetchOpsAndDelayPipeline(PrefetchInterval* currInterval, SmallVector<DataOpOptimalCycles> dataOps,
                                         size_t dataOpsCycleCost, DataOpOptimalCycles* dataOp);
        // recursively shift DMAs to previous intervals, stalling DPU pipeline for better overlap
        void maximizeOverlap(PrefetchInterval* currInterval, PrefetchInterval* targetInterval,
                             DataOpOptimalCycles dataOp);
        // try to schedule DMA during a free interval
        bool addPrefetchDataOp(DataOpOptimalCycles dataOp);

        Logger _log;
        SmallVector<PrefetchInterval> prefetchIntervals_;
    };

public:
    explicit OverlapDMAandDPU(scheduledOps& initialSchedule, AsyncDepsInfo& depsInfo);

public:
    void generatePrefetchEdgesFromOverlap(scheduleWithPrefetch& prefetchSchedule);

private:
    Logger _log;
    // incoming objects
    scheduledOps _scheduledOps;
    AsyncDepsInfo& _depsInfo;
};

}  // namespace vpux
