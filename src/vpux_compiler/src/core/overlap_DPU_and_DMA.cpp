//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/overlap_DPU_and_DMA.hpp"

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/range.hpp"

using namespace vpux;

const size_t MAX_OVERLAP_LEVEL_DIFF = 50;

//
// Constructor
//

OverlapDMAandDPU::OverlapDMAandDPU(scheduledOps& initialSchedule, AsyncDepsInfo& depsInfo)
        : _log(Logger::global().nest("chain-pipelining", 0)), _scheduledOps(initialSchedule), _depsInfo(depsInfo) {
}

//
// FreeInterval
//

OverlapDMAandDPU::FreeInterval::FreeInterval(size_t cycleBegin, size_t cycleEnd)
        : cycleBegin_(cycleBegin), cycleEnd_(cycleEnd) {
}

void OverlapDMAandDPU::FreeInterval::printFreeInterval() {
    for (auto& prefetchOp : prefetchOps_) {
        std::cout << "\t\tData op " << prefetchOp.opIdx_ << " on level " << prefetchOp.level_ << " optimal cycle from "
                  << prefetchOp.cycleBegin_ << " to " << prefetchOp.cycleEnd_ << std::endl;
    }
}

void OverlapDMAandDPU::FreeInterval::overlapActivationCopyIn(CycleInterval activationCopyInCycles, size_t copyInLevel,
                                                             bool thisIntervalActivationSpill) {
    if (thisIntervalActivationSpill) {
        // if this interval add activation copy DMA to prefetch ops
        if (prefetchOps_.empty()) {
            prefetchOps_.push_back(DataOpOptimalCycles(activationCopyInCycles.opIdx_, activationCopyInCycles.opIdx_,
                                                       copyInLevel, cycleEnd_,
                                                       cycleEnd_ + activationCopyInCycles.intervalSize()));
        } else {
            auto lastPrefetchDMA = prefetchOps_.back();
            prefetchOps_.push_back(DataOpOptimalCycles(
                    activationCopyInCycles.opIdx_, activationCopyInCycles.opIdx_, copyInLevel,
                    lastPrefetchDMA.cycleEnd_, lastPrefetchDMA.cycleEnd_ + activationCopyInCycles.intervalSize()));
        }
        cycleEnd_ += activationCopyInCycles.intervalSize();
    } else {
        // post activation operations can start earlier
        if (activationCopyInCycles.cycleEnd_ <= cycleEnd_) {
            // reduce all sizes by diff
            cycleBegin_ -= activationCopyInCycles.intervalSize();
            cycleEnd_ -= activationCopyInCycles.intervalSize();
            for (auto& p : prefetchOps_) {
                p.cycleBegin_ -= activationCopyInCycles.intervalSize();
                p.cycleEnd_ -= activationCopyInCycles.intervalSize();
            }
        }
    }
}

size_t OverlapDMAandDPU::FreeInterval::intervalSize(size_t cycleEnd) {
    return std::min(cycleEnd, cycleEnd_) - cycleBegin_;
}

void OverlapDMAandDPU::FreeInterval::addOpToFreeInterval(DataOpOptimalCycles dataOp) {
    VPUX_THROW_UNLESS(canOpFitInInterval(dataOp), "Not enough free cycles");
    // shift end cycle to end of free interval if ending during activation copy out
    auto insertedEndTime = std::min(dataOp.cycleEnd_, cycleEnd_);
    auto insertedStartTime = insertedEndTime - dataOp.cycleCost();

    // insert new data op with updated params
    prefetchOps_.push_back(
            DataOpOptimalCycles(dataOp.opIdx_, dataOp.DPUIdx_, dataOp.level_, insertedStartTime, insertedEndTime));

    auto last = prefetchOps_.rbegin();
    ++last;  // skip inserted entry
    while (last != prefetchOps_.rend()) {
        if (last->cycleEnd_ > insertedStartTime || last->cycleBegin_ > insertedEndTime) {
            // update for next iteration
            insertedEndTime = insertedStartTime;
            insertedStartTime = insertedEndTime - last->cycleCost();
            // update interval prefetch ops
            last->cycleEnd_ = insertedEndTime;
            last->cycleBegin_ = insertedStartTime;
        }
        ++last;
    }
}

void OverlapDMAandDPU::FreeInterval::addOpsToFreeIntervalWithActivationStall(
        SmallVector<OverlapDMAandDPU::DataOpOptimalCycles> dataOps) {
    // move all current ops to front
    auto front = prefetchOps_.begin();
    auto insertedStartTime = cycleBegin_;

    while (front != prefetchOps_.end()) {
        size_t insertedEndTime = insertedStartTime + front->cycleCost();
        front->cycleBegin_ = insertedStartTime;
        front->cycleEnd_ = insertedEndTime;
        insertedStartTime = insertedEndTime;
        ++front;
    }

    size_t dataOpsCycleCost = 0;
    // insert ops exceeding activation copy out
    for (auto dataOp : dataOps) {
        size_t insertedEndTime = insertedStartTime + dataOp.cycleCost();
        prefetchOps_.push_back(
                DataOpOptimalCycles(dataOp.opIdx_, dataOp.DPUIdx_, dataOp.level_, insertedStartTime, insertedEndTime));
        insertedStartTime = insertedEndTime;
        dataOpsCycleCost += dataOp.cycleCost();
    }

    // update cycle end
    cycleEnd_ = insertedStartTime;
}

void OverlapDMAandDPU::FreeInterval::addFreeIntervalStall(size_t stall) {
    // stall interval by a fixed cycle
    cycleBegin_ += stall;
    cycleEnd_ += stall;
    auto front = prefetchOps_.begin();
    while (front != prefetchOps_.end()) {
        front->cycleBegin_ += stall;
        front->cycleEnd_ += stall;
        ++front;
    }
}

void OverlapDMAandDPU::FreeInterval::addFreeIntervalStall(size_t oldCycle, size_t newCycle) {
    // stall interval by a from a cycle to a new cycle
    auto diff = newCycle - oldCycle;
    if (cycleBegin_ >= oldCycle) {
        cycleBegin_ += diff;
    }
    if (cycleEnd_ >= oldCycle) {
        cycleEnd_ += diff;
    }
    auto front = prefetchOps_.begin();
    while (front != prefetchOps_.end()) {
        if (front->cycleEnd_ >= oldCycle) {
            front->cycleBegin_ += diff;
            front->cycleEnd_ += diff;
        }
        ++front;
    }
}

SmallVector<OverlapDMAandDPU::DataOpOptimalCycles> OverlapDMAandDPU::FreeInterval::frontCandidatesOfSize(size_t size) {
    SmallVector<DataOpOptimalCycles> candidates;
    size_t candidateSize = 0;

    // retrieve candidates from front
    auto front = prefetchOps_.begin();
    while (front != prefetchOps_.end() && candidateSize < size) {
        candidateSize += front->cycleCost();
        candidates.push_back(*front);
        front = prefetchOps_.erase(front);
    }

    // reverse to preserve order since inserting from interval end
    std::reverse(candidates.begin(), candidates.end());
    return candidates;
}

bool OverlapDMAandDPU::FreeInterval::canOpFitInInterval(OverlapDMAandDPU::DataOpOptimalCycles dataOp) {
    size_t endCycle = std::min(dataOp.cycleEnd_, cycleEnd_);
    size_t intervalSubSize = endCycle - cycleBegin_;

    // account for any existing prefetch ops
    auto first = prefetchOps_.begin();
    while (first != prefetchOps_.end()) {
        if (first->cycleBegin_ < endCycle) {
            intervalSubSize -= std::min(first->cycleEnd_, endCycle) - first->cycleBegin_;
        }
        ++first;
    }
    return dataOp.cycleCost() <= intervalSubSize;
}

bool OverlapDMAandDPU::FreeInterval::canOpFitInInterval(size_t cycles) {
    return cycles <= getIntervalFreeCycles();
}

size_t OverlapDMAandDPU::FreeInterval::getIntervalFreeCycles() {
    // calculate how many cycles are not occupied by prefetch ops
    auto totalSize = cycleEnd_ - cycleBegin_;
    auto first = prefetchOps_.begin();
    while (first != prefetchOps_.end()) {
        totalSize -= first->cycleCost();
        ++first;
    }
    return totalSize;
}

//
// ActivationSpill
//

OverlapDMAandDPU::ActivationSpill::ActivationSpill(size_t cycleBegin, size_t cycleEnd, size_t level)
        : cycleBegin_(cycleBegin), cycleEnd_(cycleEnd), level_(level) {
}

bool OverlapDMAandDPU::ActivationSpill::hasActivationSpill() {
    return cycleBegin_ < cycleEnd_;
}

size_t OverlapDMAandDPU::ActivationSpill::activationSpillCycleCost() {
    return cycleEnd_ - cycleBegin_;
}

void OverlapDMAandDPU::ActivationSpill::printActivationSpill() {
    std::cout << "\tActivation copy out from " << cycleBegin_ << " to " << cycleEnd_ << std::endl;
    for (auto& write : writes_) {
        std::cout << "\t\twrite " << write.opIdx_ << " from " << write.cycleBegin_ << " to " << write.cycleEnd_
                  << std::endl;
    }
    for (auto& read : reads_) {
        std::cout << "\t\tread " << read.opIdx_ << " from " << read.cycleBegin_ << " to " << read.cycleEnd_
                  << std::endl;
    }
}

void OverlapDMAandDPU::ActivationSpill::addActivationSpillStall(size_t stall) {
    // stall interval based on a fixed cycle
    cycleBegin_ += stall;
    cycleEnd_ += stall;
    for (auto& read : reads_) {
        read.cycleBegin_ += stall;
        read.cycleEnd_ += stall;
    }
    for (auto& write : writes_) {
        write.cycleBegin_ += stall;
        write.cycleEnd_ += stall;
    }
}

bool OverlapDMAandDPU::ActivationSpill::hasReadWithIdx(operationIdxType idx) {
    // check if read exists in this interval
    for (auto& read : reads_) {
        if (read.opIdx_ == idx) {
            return true;
        }
    }
    return false;
}

void OverlapDMAandDPU::ActivationSpill::overlapActivationCopyIn(CycleInterval activationCopyInCycles,
                                                                bool thisIntervalActivationSpill) {
    if (thisIntervalActivationSpill) {
        auto first = reads_.begin();
        // find the read
        while (first != reads_.end() && first->opIdx_ != activationCopyInCycles.opIdx_) {
            ++first;
        }
        // remove that read and update all after
        first = reads_.erase(first);
        while (first != reads_.end()) {
            first->cycleBegin_ -= activationCopyInCycles.intervalSize();
            first->cycleEnd_ -= activationCopyInCycles.intervalSize();
            ++first;
        }
        cycleBegin_ += activationCopyInCycles.intervalSize();
    } else if (activationCopyInCycles.cycleEnd_ <= cycleBegin_) {
        // all post operations can execute at an earlier cycle
        for (auto& read : reads_) {
            read.cycleBegin_ -= activationCopyInCycles.intervalSize();
            read.cycleEnd_ -= activationCopyInCycles.intervalSize();
        }
        for (auto& write : writes_) {
            write.cycleBegin_ -= activationCopyInCycles.intervalSize();
            write.cycleEnd_ -= activationCopyInCycles.intervalSize();
        }
        cycleBegin_ -= activationCopyInCycles.intervalSize();
        cycleEnd_ -= activationCopyInCycles.intervalSize();
    }
}

void OverlapDMAandDPU::ActivationSpill::populateReadsAndWrites(ArrayRef<CycleInterval> reads,
                                                               ArrayRef<CycleInterval> writes) {
    // align with optimal DPU pipeline
    auto newStart = cycleBegin_;
    for (auto write : writes) {
        writes_.push_back(CycleInterval(write.opIdx_, newStart, newStart + write.intervalSize()));
        newStart += write.intervalSize();
    }
    for (auto read : reads) {
        reads_.push_back(CycleInterval(read.opIdx_, newStart, newStart + read.intervalSize()));
        newStart += read.intervalSize();
    }
    VPUX_THROW_UNLESS(newStart == cycleEnd_, "Misalignment of activation copy out read & write cycles");
}

void OverlapDMAandDPU::ActivationSpill::addActivationSpillStall(size_t oldCycle, size_t newCycle) {
    // stall interval from a cycle to a new cycle
    auto diff = newCycle - oldCycle;
    if (cycleBegin_ >= oldCycle) {
        cycleBegin_ += diff;
        cycleEnd_ += diff;
        for (auto& read : reads_) {
            read.cycleBegin_ += diff;
            read.cycleEnd_ += diff;
        }
        for (auto& write : writes_) {
            write.cycleBegin_ += diff;
            write.cycleEnd_ += diff;
        }
    }
}

//
// PrefetchInterval
//

OverlapDMAandDPU::PrefetchInterval::PrefetchInterval(OverlapDMAandDPU::FreeInterval freeInterval,
                                                     SmallVector<OverlapDMAandDPU::DPUOptimalCycles> DPUOptimalCycles,
                                                     OverlapDMAandDPU::ActivationSpill activationSpill)
        : freeInterval_(freeInterval), DPUOptimalCycles_(DPUOptimalCycles), activationSpill_(activationSpill) {
    updateDPULookUp(DPUOptimalCycles);
}

void OverlapDMAandDPU::PrefetchInterval::printPrefetchInterval() {
    std::cout << "\tFree interval from " << freeInterval_.cycleBegin_ << " to " << freeInterval_.cycleEnd_ << std::endl;
    freeInterval_.printFreeInterval();
    std::cout << "\tOptimal compute op cycles:" << std::endl;
    for (auto DPU : DPUOptimalCycles_) {
        std::cout << "\t\t" << DPU.opIdx_ << " " << DPU.cycleBegin_ << " to " << DPU.cycleEnd_ << std::endl;
    }
    activationSpill_.printActivationSpill();
}

void OverlapDMAandDPU::PrefetchInterval::overlapActivationCopyInForDPU(CycleInterval activationCopyInCycles,
                                                                       bool thisIntervalActivationSpill) {
    if (!thisIntervalActivationSpill) {
        auto first = DPUOptimalCycles_.begin();
        while (first != DPUOptimalCycles_.end()) {
            if (first->cycleBegin_ >= activationCopyInCycles.cycleEnd_) {
                // if DPUs after activation copy
                first->cycleBegin_ -= activationCopyInCycles.intervalSize();
                first->cycleEnd_ -= activationCopyInCycles.intervalSize();
            }
            ++first;
        }
    }
}

void OverlapDMAandDPU::PrefetchInterval::overlapActivationCopyIn(CycleInterval activationCopyInCycles,
                                                                 bool thisIntervalActivationSpill, size_t level) {
    freeInterval_.overlapActivationCopyIn(activationCopyInCycles, level, thisIntervalActivationSpill);
    overlapActivationCopyInForDPU(activationCopyInCycles, thisIntervalActivationSpill);
    activationSpill_.overlapActivationCopyIn(activationCopyInCycles, thisIntervalActivationSpill);
}

bool OverlapDMAandDPU::PrefetchInterval::canOpFitInInterval(OverlapDMAandDPU::DataOpOptimalCycles dataOp) {
    return freeInterval_.canOpFitInInterval(dataOp);
}

bool OverlapDMAandDPU::PrefetchInterval::canOpFitInInterval(size_t cycles) {
    return freeInterval_.canOpFitInInterval(cycles);
}

void OverlapDMAandDPU::PrefetchInterval::addOpToInterval(OverlapDMAandDPU::DataOpOptimalCycles dataOp) {
    freeInterval_.addOpToFreeInterval(dataOp);
}

size_t OverlapDMAandDPU::PrefetchInterval::updateDPUCycles() {
    // find DPU operations which have to be delayed because of input DMAs
    size_t initialEndCycle = DPUOptimalCycles_.back().cycleEnd_;
    auto firstPrefetch = freeInterval_.prefetchOps_.begin();
    while (firstPrefetch != freeInterval_.prefetchOps_.end()) {
        auto firstDPU = DPUOptimalCycles_.begin();
        while (firstDPU != DPUOptimalCycles_.end()) {
            if (firstPrefetch->level_ == firstDPU->level_) {
                if (firstPrefetch->cycleEnd_ > firstDPU->cycleBegin_) {
                    // if DPU optimally starts after, DPUs need update
                    auto newStart = firstPrefetch->cycleEnd_;
                    auto newEnd = newStart + firstDPU->cycleCost_;
                    // update all from current to end
                    auto temp = firstDPU;
                    while (temp != DPUOptimalCycles_.end()) {
                        if (newEnd > temp->cycleBegin_) {
                            temp->cycleBegin_ = newStart;
                            temp->cycleEnd_ = newEnd;
                            newStart = newEnd;
                            newEnd = newStart + temp->cycleCost_;
                        }
                        ++temp;
                    }
                }
            }
            ++firstDPU;
        }
        ++firstPrefetch;
    }
    updateDPULookUp(DPUOptimalCycles_);
    return DPUOptimalCycles_.back().cycleEnd_ - initialEndCycle;
}

size_t OverlapDMAandDPU::PrefetchInterval::addOpsToIntervalWithActivationStall(
        SmallVector<OverlapDMAandDPU::DataOpOptimalCycles> dataOps) {
    auto endCycle = activationSpill_.cycleEnd_;
    freeInterval_.addOpsToFreeIntervalWithActivationStall(dataOps);
    updateDPUCycles();
    // stall can be caused by DMAs or DPUs or both
    auto newFreeIntervalEnd = std::max(freeInterval_.prefetchOps_.back().cycleEnd_, DPUOptimalCycles_.back().cycleEnd_);
    // stall activation copy out and update free interval end
    freeInterval_.cycleEnd_ = newFreeIntervalEnd;
    activationSpill_.addActivationSpillStall(newFreeIntervalEnd - activationSpill_.cycleBegin_);
    return activationSpill_.cycleEnd_ - endCycle;
}

void OverlapDMAandDPU::PrefetchInterval::updateDPULookUp(
        ArrayRef<OverlapDMAandDPU::DPUOptimalCycles> DPUOptimalCycles) {
    optimalDPUStartTime_.clear();
    for (auto& DPUCycles : DPUOptimalCycles) {
        // populate look-up table
        optimalDPUStartTime_[DPUCycles.opIdx_] = DPUCycles.cycleBegin_;
    }
}

void OverlapDMAandDPU::PrefetchInterval::addDPUStall(size_t stall) {
    // stall all DPU operations
    auto first = DPUOptimalCycles_.begin();
    while (first != DPUOptimalCycles_.end()) {
        first->cycleBegin_ += stall;
        first->cycleEnd_ += stall;
        ++first;
    }
    updateDPULookUp(DPUOptimalCycles_);
}

void OverlapDMAandDPU::PrefetchInterval::addDPUStall(size_t oldCycle, size_t newCycle) {
    auto diff = newCycle - oldCycle;
    auto first = DPUOptimalCycles_.begin();
    while (first != DPUOptimalCycles_.end()) {
        if (first->cycleEnd_ > oldCycle) {
            first->cycleBegin_ += diff;
            first->cycleEnd_ += diff;
        }
        ++first;
    }
    updateDPULookUp(DPUOptimalCycles_);
}

void OverlapDMAandDPU::PrefetchInterval::stallIntervals(size_t stall) {
    // stall based on a fixed stall
    freeInterval_.addFreeIntervalStall(stall);
    addDPUStall(stall);
    activationSpill_.addActivationSpillStall(stall);
}

void OverlapDMAandDPU::PrefetchInterval::stallIntervals(size_t oldCycle, size_t newCycle) {
    // stall based on cycles
    freeInterval_.addFreeIntervalStall(oldCycle, newCycle);
    addDPUStall(oldCycle, newCycle);
    activationSpill_.addActivationSpillStall(oldCycle, newCycle);
}

SmallVector<OverlapDMAandDPU::DataOpOptimalCycles> OverlapDMAandDPU::PrefetchInterval::getCandidatesGreaterThan(
        size_t size) {
    return freeInterval_.frontCandidatesOfSize(size);
}

size_t OverlapDMAandDPU::PrefetchInterval::freeIntervalTotalSize(size_t cycleEnd) {
    return freeInterval_.intervalSize(cycleEnd);
}

size_t OverlapDMAandDPU::PrefetchInterval::getIntervalStart() {
    return freeInterval_.cycleBegin_;
}

size_t OverlapDMAandDPU::PrefetchInterval::getIntervalEnd() {
    return activationSpill_.cycleEnd_;
}

ArrayRef<OverlapDMAandDPU::DataOpOptimalCycles>
OverlapDMAandDPU::PrefetchInterval::getDataOpCyclesDuringFreeInterval() {
    return freeInterval_.prefetchOps_;
}

ArrayRef<OverlapDMAandDPU::DPUOptimalCycles> OverlapDMAandDPU::PrefetchInterval::getDPUCyclesDuringPrefetchInterval() {
    return DPUOptimalCycles_;
}

bool OverlapDMAandDPU::PrefetchInterval::isDPUExecutingDuringInterval(operationIdxType opIdx) {
    return optimalDPUStartTime_.find(opIdx) != optimalDPUStartTime_.end();
}

size_t OverlapDMAandDPU::PrefetchInterval::getDPUBeginCycle(operationIdxType opIdx) {
    return optimalDPUStartTime_[opIdx];
}

//
// PrefetchPipeline
//

OverlapDMAandDPU::PrefetchPipeline::PrefetchPipeline(): _log(Logger::global().nest("prefetch-pipeline", 0)) {
}

void OverlapDMAandDPU::PrefetchPipeline::addPrefetchInterval(PrefetchInterval interval) {
    prefetchIntervals_.push_back(interval);
}

SmallVector<OverlapDMAandDPU::ActivationSpill> OverlapDMAandDPU::PrefetchPipeline::getActivationSpills() {
    SmallVector<ActivationSpill> activationSpills;
    for (auto& prefetchInterval : prefetchIntervals_) {
        activationSpills.push_back(prefetchInterval.activationSpill_);
    }
    return activationSpills;
}

void OverlapDMAandDPU::PrefetchPipeline::overlapActivationCopyIn(CycleInterval activationCopyInCycles, size_t level) {
    // remove the stall from activation copy-in by moving the copy to the previous
    // free interval, thus shifting the entire DMA pipeline so that the activation
    // copy-in overlaps with the previous DPU and update cycles for later operations
    for (auto& prefetchInterval : prefetchIntervals_) {
        bool thisIntervalActivationSpill =
                prefetchInterval.activationSpill_.hasReadWithIdx(activationCopyInCycles.opIdx_);
        prefetchInterval.overlapActivationCopyIn(activationCopyInCycles, thisIntervalActivationSpill, level);
    }
}

void OverlapDMAandDPU::PrefetchPipeline::printPrefetchPipeline() {
    for (auto& prefetchInterval : prefetchIntervals_) {
        std::cout << "Prefetch interval from " << prefetchInterval.getIntervalStart() << " to "
                  << prefetchInterval.getIntervalEnd() << std::endl;
        prefetchInterval.printPrefetchInterval();
    }
}

void OverlapDMAandDPU::PrefetchPipeline::stallPipeline(size_t oldCycle, size_t newCycle) {
    for (auto& prefetchInterval : prefetchIntervals_) {
        prefetchInterval.stallIntervals(oldCycle, newCycle);
    }
}

size_t OverlapDMAandDPU::PrefetchPipeline::getOptimalDPUBeginCycle(operationIdxType opIdx) {
    for (auto& prefetchInterval : prefetchIntervals_) {
        if (prefetchInterval.isDPUExecutingDuringInterval(opIdx)) {
            return prefetchInterval.getDPUBeginCycle(opIdx);
        }
    }
    VPUX_THROW("Error DPU not found in prefetch intervals");
}

void OverlapDMAandDPU::PrefetchPipeline::updateDMAOptimalCycles(OverlapDMAandDPU::DataOpOptimalCycles* dataOp) {
    // update dataOp start time
    size_t newEndTime = getOptimalDPUBeginCycle(dataOp->DPUIdx_);
    if (newEndTime < dataOp->cycleCost()) {
        // operation can not finish earlier than its cycle cost
        newEndTime = dataOp->cycleCost();
    }
    auto newStartTime = newEndTime - dataOp->cycleCost();

    dataOp->cycleBegin_ = newStartTime;
    dataOp->cycleEnd_ = newEndTime;
}

SmallVector<OverlapDMAandDPU::DPUOptimalCycles> OverlapDMAandDPU::PrefetchPipeline::getOptimalDPUCycles() {
    SmallVector<DPUOptimalCycles> optimalDPUCycles;
    for (auto& prefetchInterval : prefetchIntervals_) {
        auto intervalCycles = prefetchInterval.getDPUCyclesDuringPrefetchInterval();
        optimalDPUCycles.append(intervalCycles.begin(), intervalCycles.end());
    }
    return optimalDPUCycles;
}

void OverlapDMAandDPU::PrefetchPipeline::populatePrefetchEdges(scheduleWithPrefetch& prefetchSchedule) {
    // all operations should have been added to an interval
    _log.trace("Populate prefetch edges");
    for (auto& prefetchInterval : prefetchIntervals_) {
        _log.nest().trace("prefetch interval {0} to {1}", prefetchInterval.getIntervalStart(),
                          prefetchInterval.getIntervalEnd());
        _log.nest().trace("free interval {0} to {1}", prefetchInterval.freeInterval_.cycleBegin_,
                          prefetchInterval.freeInterval_.cycleEnd_);
        _log.nest().trace("activation copy out interval {0} to {1}", prefetchInterval.activationSpill_.cycleBegin_,
                          prefetchInterval.activationSpill_.cycleEnd_);
        // check if this prefetch interval contains activation reads
        bool activationSpill = prefetchInterval.activationSpill_.reads_.empty();
        auto DPUCycles = prefetchInterval.getDPUCyclesDuringPrefetchInterval();
        for (auto& DPU : DPUCycles) {
            _log.nest(2).trace("DPU {0} with cycles {1} to {2} can prefetch:", DPU.opIdx_, DPU.cycleBegin_,
                               DPU.cycleEnd_);
            prefetchSet newSet;
            for (auto DMA : prefetchInterval.getDataOpCyclesDuringFreeInterval()) {
                if (DMA.level_ > DPU.level_ && DMA.cycleBegin_ <= DPU.cycleEnd_ && DMA.cycleBegin_ >= DPU.cycleBegin_) {
                    // link future DMA to DPU
                    _log.nest(3).trace("DMA {0} with cycles {1} to {2}", DMA.opIdx_, DMA.cycleBegin_, DMA.cycleEnd_);
                    newSet.insert(PrefetchDMA(DMA.opIdx_, DMA.level_));
                }
            }
            // and also if current DPU is the last DPU after which the activation spill will occur
            activationSpill = (activationSpill && DPU.opIdx_ == DPUCycles.back().opIdx_);
            prefetchSchedule.push_back(OverlappedSchedule(DPU.opIdx_, DPU.level_, newSet, activationSpill));
        }
    }
}

OverlapDMAandDPU::PrefetchInterval* OverlapDMAandDPU::PrefetchPipeline::findTargetInterval(size_t cycleEnd) {
    // want interval where data op can complete before activation copy out
    auto firstInterval = prefetchIntervals_.begin();
    auto lastFreeInterval = firstInterval;
    while (firstInterval != prefetchIntervals_.end()) {
        if (firstInterval->getIntervalStart() < cycleEnd && cycleEnd <= firstInterval->getIntervalEnd()) {
            return firstInterval;
        }
        if (firstInterval->getIntervalEnd() < cycleEnd) {
            lastFreeInterval = firstInterval;
        }
        ++firstInterval;
    }
    // data bound part of network
    return lastFreeInterval;
}

bool OverlapDMAandDPU::PrefetchPipeline::shiftToPreviousInterval(PrefetchInterval* currentInterval,
                                                                 OverlapDMAandDPU::DataOpOptimalCycles dataOp) {
    // try to insert in previous interval
    auto newTargetInterval = currentInterval;

    // introduce limit for recursion of long chains
    auto intervalEndLevel = currentInterval->getDPUCyclesDuringPrefetchInterval().back().level_;
    if (dataOp.level_ >= intervalEndLevel && dataOp.level_ - intervalEndLevel > MAX_OVERLAP_LEVEL_DIFF) {
        _log.nest(5).trace("Level difference limit reached with DMA {0}", dataOp.opIdx_);
        return false;
    }

    if (newTargetInterval->canOpFitInInterval(dataOp)) {
        _log.nest(5).trace("DMA {0} scheduled during previous interval", dataOp.opIdx_);
        newTargetInterval->addOpToInterval(dataOp);
        return true;
    } else if (newTargetInterval != prefetchIntervals_.begin()) {
        // need to move DMAs back
        _log.nest(5).trace("DMA {0} NOT scheduled during previous interval", dataOp.opIdx_);
        size_t missingSpace = dataOp.cycleCost() - newTargetInterval->freeInterval_.getIntervalFreeCycles();
        auto candidatesToMakeSpace = newTargetInterval->getCandidatesGreaterThan(missingSpace);
        auto prevInterval = newTargetInterval;
        --prevInterval;
        for (auto can : candidatesToMakeSpace) {
            _log.nest(6).trace("need to move {0} of size {1} to previous interval", can.opIdx_, can.cycleCost());
            if (!shiftToPreviousInterval(prevInterval, can)) {
                return false;
            }
        }
        // candidates should have been shifed now
        if (newTargetInterval->canOpFitInInterval(dataOp)) {
            _log.nest(5).trace("space made and DMA {0} scheduled during previous interval", dataOp.opIdx_);
            newTargetInterval->addOpToInterval(dataOp);
            return true;
        } else {
            // move back to previous interval until the front is reached
            _log.nest(5).trace("DMA {0} will try to be scheduled during previous interval", dataOp.opIdx_);
            return shiftToPreviousInterval(prevInterval, dataOp);
        }
    }
    // first interval - can not shift back
    _log.nest(5).trace("DMA {0} can not shift back", dataOp.opIdx_);
    return false;
}

void OverlapDMAandDPU::PrefetchPipeline::prefetchOpsAndDelayPipeline(
        PrefetchInterval* currInterval, SmallVector<OverlapDMAandDPU::DataOpOptimalCycles> dataOps,
        size_t dataOpsCycleCost, OverlapDMAandDPU::DataOpOptimalCycles* dataOp) {
    // case for a different interval
    if (currInterval->canOpFitInInterval(dataOpsCycleCost)) {
        for (auto dataOp : dataOps) {
            currInterval->addOpToInterval(dataOp);
        }
    } else {
        // add ops to interval and delay everything!!!
        size_t totalPipelineDelay = currInterval->addOpsToIntervalWithActivationStall(dataOps);
        _log.nest(6).trace("delaying pipeline by {0}", totalPipelineDelay);
        // delay pipeline - all following prefetch intervals
        auto temp = currInterval;
        ++temp;  // skip currInterval
        while (temp != prefetchIntervals_.end()) {
            temp->stallIntervals(totalPipelineDelay);
            ++temp;
        }
        updateDMAOptimalCycles(dataOp);
    }
}

void OverlapDMAandDPU::PrefetchPipeline::maximizeOverlap(PrefetchInterval* currInterval,
                                                         PrefetchInterval* targetInterval,
                                                         OverlapDMAandDPU::DataOpOptimalCycles dataOp) {
    if (currInterval == targetInterval) {
        // case for target interval - op has to be added here
        _log.nest(5).trace("add to current interval {0}", dataOp.opIdx_);

        if (currInterval->canOpFitInInterval(dataOp)) {
            // if operation can fit - simply add
            currInterval->addOpToInterval(dataOp);
        } else {
            // shift interval operations operations and insert the prefetch op at the end
            SmallVector<DataOpOptimalCycles> dataOps{dataOp};
            // add ops to interval and delay everything!!!
            size_t totalPipelineDelay = currInterval->addOpsToIntervalWithActivationStall(dataOps);
            _log.nest(6).trace("delaying pipeline by {0}", totalPipelineDelay);
            // delay pipeline - all following prefetch intervals
            auto temp = currInterval;
            ++temp;  // skip currInterval
            while (temp != prefetchIntervals_.end()) {
                temp->stallIntervals(totalPipelineDelay);
                ++temp;
            }
        }

        return;
    }

    auto nextInterval = currInterval;
    ++nextInterval;
    // retrieve interval cycle gaps - free cycles
    size_t currIntervalFreeCycles = currInterval->freeInterval_.getIntervalFreeCycles();

    if (currIntervalFreeCycles > 0) {
        // try to fill the free cycles with operations from next intervals
        _log.nest(5).trace("try to fill the free cycles with operations from next intervals {0}, cycles left {1}",
                           dataOp.opIdx_, currIntervalFreeCycles);
        auto temp = nextInterval;
        SmallVector<DataOpOptimalCycles> dataOps;
        size_t totalSize = 0;
        // get candidates of required size from next intervals
        while (totalSize < currIntervalFreeCycles && temp != prefetchIntervals_.end()) {
            auto nextIntervalOps = temp->getCandidatesGreaterThan(currIntervalFreeCycles - totalSize);
            // move candidates to previous interval
            for (auto& op : nextIntervalOps) {
                _log.nest(6).trace("moving DMA {0} to prev interval", op.opIdx_);
                totalSize += op.cycleCost();
            }
            dataOps.append(nextIntervalOps.begin(), nextIntervalOps.end());
            ++temp;
        }
        if (totalSize < currIntervalFreeCycles) {
            // if candidates do not exceed interval size add the prefetch op to
            // make the interval data bound
            dataOps.push_back(dataOp);
            totalSize += dataOp.cycleCost();
            // make interval data bound and delay pipeline
            prefetchOpsAndDelayPipeline(currInterval, dataOps, totalSize, &dataOp);
            return;
        } else {
            // do not add prefetch op but move candidates to make interval data bound
            prefetchOpsAndDelayPipeline(currInterval, dataOps, totalSize, &dataOp);
            // find new target and try to add data op again
            targetInterval = findTargetInterval(dataOp.cycleEnd_);
            if (targetInterval->canOpFitInInterval(dataOp)) {
                // check if now target interval has space for op
                _log.nest(5).trace("target interval has space for {0}", dataOp.opIdx_);
                targetInterval->addOpToInterval(dataOp);
                return;
            }
        }
    }

    // move to next interval
    _log.nest(5).trace("move to next interval with DMA {0}", dataOp.opIdx_);
    maximizeOverlap(nextInterval, targetInterval, dataOp);
}

bool OverlapDMAandDPU::PrefetchPipeline::addPrefetchDataOp(OverlapDMAandDPU::DataOpOptimalCycles dataOp) {
    // find target insertion point - end cycle
    auto targetInterval = findTargetInterval(dataOp.cycleEnd_);
    _log.nest(2).trace("target interval found {0} to {1}", targetInterval->getIntervalStart(),
                       targetInterval->getIntervalEnd());

    if (targetInterval->canOpFitInInterval(dataOp)) {
        // simple case - add op to interval
        _log.nest(3).trace("simple case, DMA {0} scheduled during optimal cycles", dataOp.opIdx_);
        targetInterval->addOpToInterval(dataOp);
    } else {
        // complex case
        _log.nest(3).trace("complex case, DMA {0}", dataOp.opIdx_);

        // save backup in case of failure
        auto backUpIntervals = prefetchIntervals_;
        _log.nest(4).trace("try to shift to previous interval");

        // try to shift DMAs
        // Note might change DMA order without check
        // targetInterval->freeIntervalTotalSize(dataOp.cycleEnd_) >= dataOp.cycleCost()
        if (shiftToPreviousInterval(targetInterval, dataOp)) {
            _log.nest(5).trace("success shifting to previous interval");
            return true;
        }
        _log.nest(5).trace("failure shifting to previous interval");
        prefetchIntervals_ = backUpIntervals;

        _log.nest(4).trace("try to maximize overlap by delaying DPU start time");
        // try to fill all leftover cycles with exceeding into activation copy out
        // NOTE: will delay compute start time
        maximizeOverlap(prefetchIntervals_.begin(), targetInterval, dataOp);
    }
    return true;
}

void OverlapDMAandDPU::generatePrefetchEdgesFromOverlap(scheduleWithPrefetch& prefetchSchedule) {
    _log.trace("Creating pipelining chains");

    // TODO
    // create optimal compute op order
    // use that order with the following algorithm

    auto op = _scheduledOps.begin();
    SmallVector<ScheduledOpInfo*> computeOps;
    mlir::DenseMap<operationIdxType, size_t> computeOpLevel;
    mlir::DenseMap<operationIdxType, ScheduledOpInfo*> computeOpOnLevel;
    mlir::DenseMap<operationIdxType, size_t> dataOpCycleCosts;
    mlir::DenseMap<operationIdxType, size_t> dataOpLevel;
    mlir::DenseMap<size_t, SmallVector<ScheduledOpInfo*>> candidatesOnLevel;
    mlir::DenseMap<size_t, ActivationSpillCost> activationSpills;

    // STEP 1: Representation

    size_t currentDPUCycle = 1;
    size_t currentLevel = 0;
    activationSpills[currentLevel] = ActivationSpillCost(currentLevel);
    _log.trace("Finding Intervals");

    // find intervals
    while (op != _scheduledOps.end()) {
        if (!op->isOriginalOp() || op->isNonComputeChain) {
            // skip implicit spill and profiling operations
            ++op;
            continue;
        }
        size_t operationCycleCost = op->cycleEnd_ - op->cycleBegin_;
        if (op->executor == VPU::ExecutorKind::SHAVE_UPA || op->executor == VPU::ExecutorKind::SHAVE_ACT ||
            op->executor == VPU::ExecutorKind::SHAVE_NN || op->executor == VPU::ExecutorKind::DPU) {
            // currently not supported executors
        } else if (op->executor == VPU::ExecutorKind::DMA_NN) {
            // DMA case
            if (op->isDataOp_) {
                // copy in from DDR to NNCMX
                if (_depsInfo.getOpDeps(op->op_).empty()) {
                    // prefetchable op - store DMAs which can be prefetched
                    dataOpCycleCosts[op->op_] = operationCycleCost;
                    dataOpLevel[op->op_] = currentLevel;
                    candidatesOnLevel[currentLevel].push_back(op);
                } else {
                    // activation copy in - do not delay - can not prefetch at this cycle
                    activationSpills[currentLevel].addActivationSpillRead(op->op_, op->cycleBegin_, op->cycleEnd_);
                    _log.nest().trace("activation copy out read level {0} from {1} to {2}", currentLevel,
                                      op->cycleBegin_, op->cycleEnd_);
                    currentDPUCycle += operationCycleCost;
                }
            } else {
                // copy out from NNCMX to DDR - do not delay - can not prefetch at this cycle
                activationSpills[currentLevel].addActivationSpillWrite(op->op_, op->cycleBegin_, op->cycleEnd_);
                _log.nest().trace("activation copy out write level {0} from {1} to {2}", currentLevel, op->cycleBegin_,
                                  op->cycleEnd_);
                currentDPUCycle += operationCycleCost;
            }
        } else if (op->executor == VPU::ExecutorKind::NCE) {
            // DPU case
            // store info
            computeOps.push_back(op);
            computeOpLevel[op->op_] = currentLevel;
            computeOpOnLevel[currentLevel] = op;
            // increase levels
            currentLevel++;
            // update cycles
            currentDPUCycle = op->cycleEnd_;
            // create activation copy out for next level
            activationSpills[currentLevel] = ActivationSpillCost(currentLevel);
        } else {
            VPUX_THROW("Undefined executor");
        }
        ++op;
    }

    // STEP 2: create optimal compute pipeline

    PrefetchPipeline prefetchPipeline;

    // pipelines start from 1
    size_t DMACycles = 1;
    size_t DPUCycles = 1;

    SmallVector<DPUOptimalCycles> orderedOptimalComputeOpCycles;
    _log.trace("Finding optimal compute pipeline");

    // create a schedule of only compute operations and activation copy out DMAs
    for (auto compute : computeOps) {
        VPUX_THROW_UNLESS(computeOpLevel.find(compute->op_) != computeOpLevel.end(),
                          "Cannot find op : {0} in computeOpLevel DenseMap", compute->op_);
        auto computeLevel = computeOpLevel[compute->op_];

        if (computeLevel == 0) {
            // has to wait for input, use original start time
            DPUCycles = compute->cycleBegin_;
            DMACycles = compute->cycleBegin_;
        }

        auto cycleCost = compute->cycleEnd_ - compute->cycleBegin_;
        orderedOptimalComputeOpCycles.push_back(
                DPUOptimalCycles(compute->op_, computeLevel, DPUCycles, DPUCycles + cycleCost));

        // update DPU cycles with current compute op
        DPUCycles += cycleCost;
        // check if activation copy out
        if (activationSpills[computeLevel + 1].hasActivationSpill()) {
            _log.nest().trace("new interval created to level {0}", computeLevel);
            // store intervals
            _log.nest(2).trace("free interval {0} to {1}", DMACycles, DPUCycles);
            FreeInterval freeInterval(DMACycles, DPUCycles);
            _log.nest(2).trace("apill interval {0} to {1}", DPUCycles,
                               DPUCycles + activationSpills[computeLevel + 1].activationSpillCycleCost());
            ActivationSpill activationSpill(DPUCycles,
                                            DPUCycles + activationSpills[computeLevel + 1].activationSpillCycleCost(),
                                            computeLevel + 1);
            activationSpill.populateReadsAndWrites(activationSpills[computeLevel + 1].reads_,
                                                   activationSpills[computeLevel + 1].writes_);
            // populate intervals
            prefetchPipeline.addPrefetchInterval(
                    PrefetchInterval(freeInterval, orderedOptimalComputeOpCycles, activationSpill));
            // start next compute right after activation copy out
            DPUCycles += activationSpills[computeLevel + 1].activationSpillCycleCost();
            DMACycles = DPUCycles;
            orderedOptimalComputeOpCycles.clear();
        }
    }

    _log.trace("Optimal DPU pipeline");
    for (auto optimalComputeOp : prefetchPipeline.getOptimalDPUCycles()) {
        _log.nest().trace("optimal compute {0} start at {1} end {2} ", optimalComputeOp.opIdx_,
                          optimalComputeOp.cycleBegin_, optimalComputeOp.cycleEnd_);
    }

    // STEP 3: find optimal cycle start time for DMAs
    _log.trace("Finding Optimal DMA pipeline");

    for (size_t level = 0; level < candidatesOnLevel.size(); level++) {
        // skip if no compute on this level
        if (computeOpOnLevel.find(level) == computeOpOnLevel.end()) {
            continue;
        }
        SmallVector<ScheduledOpInfo*> dataOpsOnLevel(candidatesOnLevel[level]);
        // sort based on number of cycles to prefetch larger operations first
        // to minimize fragmentation and have a larger benefit for prefetch
        llvm::sort(dataOpsOnLevel.begin(), dataOpsOnLevel.end(),
                   [](const ScheduledOpInfo* val1, const ScheduledOpInfo* val2) {
                       return (val1->cycleEnd_ - val1->cycleBegin_) > (val2->cycleEnd_ - val2->cycleBegin_);
                   });

        auto computeOnLevel = computeOpOnLevel[level];
        // ensure all DMAs will execute before optimalComputeStartTime
        for (auto dataOp : dataOpsOnLevel) {
            auto optimalComputeStartTime = prefetchPipeline.getOptimalDPUBeginCycle(computeOnLevel->op_);
            auto optimalEndTime = optimalComputeStartTime;
            if (optimalEndTime < dataOpCycleCosts[dataOp->op_]) {
                prefetchPipeline.stallPipeline(optimalEndTime, dataOpCycleCosts[dataOp->op_]);
                optimalEndTime = dataOpCycleCosts[dataOp->op_];
            }
            auto optimalStartTime = optimalEndTime - dataOpCycleCosts[dataOp->op_];

            if (level > 0) {
                _log.nest().trace("DMA {0} of size {1} for compute {2} requires allocation from {3} to {4}",
                                  dataOp->op_, dataOpCycleCosts[dataOp->op_], computeOnLevel->op_, optimalStartTime,
                                  optimalEndTime);
                prefetchPipeline.addPrefetchDataOp(DataOpOptimalCycles(
                        dataOp->op_, computeOnLevel->op_, dataOpLevel[dataOp->op_], optimalStartTime, optimalEndTime));
            }
        }
    }

    // STEP 4: overlap ActivationSpills with DPU
    _log.trace("Overlap activation copy outs with DPU");

    std::set<operationIdxType> executedWrites;
    for (auto activationSpill : prefetchPipeline.getActivationSpills()) {
        if (activationSpill.hasActivationSpill()) {
            for (auto& read : activationSpill.reads_) {
                auto writesExecuted = true;
                for (auto dep : _depsInfo.getOpDeps(read.opIdx_)) {
                    if (executedWrites.find(dep) == executedWrites.end()) {
                        writesExecuted = false;
                        break;
                    }
                }
                if (writesExecuted) {
                    // overlap only if writes executed previously
                    _log.nest().trace("Overlap AS {0} from cycles {1} to {2} with level {3}", read.opIdx_,
                                      read.cycleBegin_, read.cycleEnd_, activationSpill.level_);
                    prefetchPipeline.overlapActivationCopyIn(read, activationSpill.level_);
                }
            }
            for (auto& write : activationSpill.writes_) {
                executedWrites.insert(write.opIdx_);
            }
        }
    }

    // Debug option to print the pipeline
    // prefetchPipeline.printPrefetchPipeline();

    // STEP 5: link DMA to DPU
    _log.trace("Link DMA to DPU");

    prefetchPipeline.populatePrefetchEdges(prefetchSchedule);

    // TODO

    // use allocator with information about alive periods of tensors to
    // allocate CMX memory (and DDR?) for the buffers
    // optimal solution with minimal spilling

    // post analysis of where the spills and stalls occur
    // return the cycle cost of pipeline with information about activation copy outs
    // not removed, reduce the appropriate layers size and try to get a better
    // feedback from the loop
}
