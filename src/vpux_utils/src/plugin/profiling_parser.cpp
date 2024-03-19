//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/plugin/profiling_parser.hpp"

#include "vpux/utils/core/profiling.hpp"
#include "vpux/utils/plugin/profiling_meta.hpp"

#include <schema/profiling_generated.h>

#include <cstring>
#include <map>
#include <sstream>
#include <string>

using namespace vpux::profiling;

// Main synchronization primitive. First contain tasks, that update barrier, Second - that wait for barrier
using SynchronizationPoint = std::pair<RawProfilingRecords, RawProfilingRecords>;
using SynchronizationPointsContainer = std::vector<SynchronizationPoint>;

enum class SynchronizationPointKind { DMA_TO_DPU, DPU_TO_DMA, DMA_TO_UPA, STRICT_DMA_TO_DPU };

namespace {

constexpr double PROF_CLK_37XX_DEFAULT_VALUE_MHZ = 38.4;
struct Dma27BandwidthProvider {
    static constexpr double value = Dma27Bandwidth;
};

struct Dma40BandwidthProvider {
    static constexpr double value = Dma40Bandwidth;
};

template <typename BandwidthProvider, bool SHARED_DMA_SW_CNT, bool SHARED_DMA_DPU_CNT>
constexpr FrequenciesSetup getFreqSetupHelper(const double vpuFreq, const double dpuFreq, double profClk) {
    return FrequenciesSetup{vpuFreq, dpuFreq, profClk, BandwidthProvider::value, SHARED_DMA_SW_CNT, SHARED_DMA_DPU_CNT};
}

constexpr auto getFreqSetup37XXHelper = getFreqSetupHelper<Dma27BandwidthProvider, true, false>;

FrequenciesSetup getFpgaFreqSetup(MVCNN::TargetDevice device) {
    VPUX_THROW("TargetDevice {0} is not supported ", MVCNN::EnumNameTargetDevice(device));
}

FrequenciesSetup getFreqSetupFromPll(MVCNN::TargetDevice device, uint16_t pll, bool /*highFreqPerfClk*/) {
    switch (device) {
    case MVCNN::TargetDevice::TargetDevice_VPUX37XX:
        return FrequenciesSetup::get37XXSetup(pll);
    default:
        VPUX_THROW("TargetDevice {0} is not supported ", MVCNN::EnumNameTargetDevice(device));
    }
}

template <class IterableContainer>
std::string convertIterableToString(const IterableContainer& container) {
    if (container.empty()) {
        return "[]";
    }
    const auto last = std::prev(container.cend());
    std::stringstream ss;
    ss << "[";
    for (auto it = container.cbegin(); it != last; ++it) {
        ss << *it << ", ";
    }
    ss << *last << "]";
    return ss.str();
}

inline std::string to_string(const SynchronizationPoint& syncPoint) {
    const auto getType = [](const auto& vec) -> std::string {
        return convertExecTypeToName(vec.front()->getExecutorType());
    };
    const auto printNames = [](const auto& x) {
        std::vector<std::string> names;
        for (const auto& t : x) {
            names.push_back(t->getTaskName());
        }
        return convertIterableToString(names);
    };

    return printNames(syncPoint.first) + " " + getType(syncPoint.first) + " -> " + getType(syncPoint.second) + " " +
           printNames(syncPoint.second);
}

template <typename... Args>
void warnOrFail(bool failOnError, vpux::Logger& log, bool condition, llvm::StringLiteral format, Args&&... params) {
    if (condition) {
        return;
    }
    if (failOnError) {
        VPUX_THROW(format, std::forward<Args>(params)...);
    } else {
        log.warning(format, std::forward<Args>(params)...);
    }
}

RawProfilingRecords getRelatedTasksOfKind(
        RawProfilingRecord::BarrierIdType barrierId,
        const std::multimap<RawProfilingRecord::BarrierIdType, RawProfilingRecordPtr>& relatedTasks,
        ExecutorType execKind) {
    RawProfilingRecords tasks;
    auto range = relatedTasks.equal_range(barrierId);
    for (auto it = range.first; it != range.second; ++it) {
        if (it->second->getExecutorType() == execKind) {
            tasks.push_back(it->second);
        }
    }
    return tasks;
}

SynchronizationPointsContainer findSynchronizationPoints(const RawProfilingRecords& taskGroup1,
                                                         const RawProfilingRecords& taskGroup2,
                                                         SynchronizationPointKind pointKind) {
    using BarrierIdType = RawProfilingRecord::BarrierIdType;
    std::multimap<BarrierIdType, RawProfilingRecordPtr> barrierPredecessors;
    std::set<BarrierIdType> waitBarriers;
    std::multimap<BarrierIdType, RawProfilingRecordPtr> barrierSuccessors;
    std::set<BarrierIdType> updateBarriers;
    std::unordered_set<BarrierIdType> dpuUpdatedBarriers;

    for (const auto& tasksGroup : {taskGroup1, taskGroup2}) {
        for (const auto& task : tasksGroup) {
            for (const auto waitBarrier : task->getWaitBarriers()) {
                barrierSuccessors.insert(std::make_pair(waitBarrier, task));
                waitBarriers.insert(waitBarrier);
            }
            for (const auto updateBarrier : task->getUpdateBarriers()) {
                barrierPredecessors.insert(std::make_pair(updateBarrier, task));
                updateBarriers.insert(updateBarrier);
                if (task->getExecutorType() == ExecutorType::DPU) {
                    dpuUpdatedBarriers.insert(updateBarrier);
                }
            }
        }
    }

    ExecutorType predecessorExecType = ExecutorType::DMA_SW;
    ExecutorType successorExecType = ExecutorType::DPU;
    if (pointKind == SynchronizationPointKind::DPU_TO_DMA) {
        std::swap(predecessorExecType, successorExecType);
    } else if (pointKind == SynchronizationPointKind::DMA_TO_UPA) {
        successorExecType = ExecutorType::UPA;
    }

    // Possible synchronization points occurs on covered from both directions barriers
    const auto commonBarriers = RawProfilingRecord::getBarriersIntersection(waitBarriers, updateBarriers);
    SynchronizationPointsContainer synchronizationPoints;
    size_t numUnsuitableSyncPoints = 0;
    for (const auto& commonBarrier : commonBarriers) {
        if (pointKind == SynchronizationPointKind::STRICT_DMA_TO_DPU && dpuUpdatedBarriers.count(commonBarrier) != 0) {
            ++numUnsuitableSyncPoints;
            continue;
        }
        RawProfilingRecords predecessors =
                getRelatedTasksOfKind(commonBarrier, barrierPredecessors, predecessorExecType);
        RawProfilingRecords successors = getRelatedTasksOfKind(commonBarrier, barrierSuccessors, successorExecType);
        if (!predecessors.empty() && !successors.empty()) {
            synchronizationPoints.push_back(std::make_pair(predecessors, successors));
        }
    }
    if (numUnsuitableSyncPoints != 0) {
        vpux::Logger::global().trace(
                "Found {0} synchronization points. {1} are unused because of requested SynchronizationPointKind",
                synchronizationPoints.size(), numUnsuitableSyncPoints);
    }
    return synchronizationPoints;
}

// Get a shift in time for the synchronization point. The shift is defined as a difference between the latest update
// task and the earliest wait task
std::vector<RawProfilingRecord::TimeType> getBarrierShiftEstimations(const SynchronizationPointsContainer& syncPoints,
                                                                     FrequenciesSetup frequenciesSetup,
                                                                     vpux::Logger& log, bool extraVerbosity = false) {
    using TimeType = RawProfilingRecord::TimeType;

    std::vector<double> syncShiftsEstimations;
    for (const auto& syncPoint : syncPoints) {
        TimeType latestPreBarrierTask = std::numeric_limits<TimeType>::min();
        const auto predecessors = syncPoint.first;
        for (const auto& predecessor : predecessors) {
            latestPreBarrierTask = std::max(latestPreBarrierTask, predecessor->getFinishTime(frequenciesSetup));
        }

        TimeType earliestPostBarrierTask = std::numeric_limits<TimeType>::max();
        const auto successors = syncPoint.second;
        for (const auto& successor : successors) {
            earliestPostBarrierTask = std::min(earliestPostBarrierTask, successor->getStartTime(frequenciesSetup));
        }

        const auto shiftEstimation = latestPreBarrierTask - earliestPostBarrierTask;
        syncShiftsEstimations.push_back(shiftEstimation);
        if (extraVerbosity) {
            log.trace("{0}", to_string(syncPoint));
            log.trace(" {0} - {1} = {2}", latestPreBarrierTask, earliestPostBarrierTask, shiftEstimation);
        }
    }
    return syncShiftsEstimations;
}

// Function return difference as delta = DMA - other, so later we can just add delta to convert from Other(DPU/UPA) to
// DMA timer
std::optional<int64_t> getDMA2OtherTimersShift(const RawProfilingRecords& dmaTasks,
                                               const RawProfilingRecords& otherTasks, FrequenciesSetup frequenciesSetup,
                                               SynchronizationPointKind pointKind, vpux::Logger& log) {
    const auto inverseAlgorithm = pointKind == SynchronizationPointKind::DPU_TO_DMA;
    // For some reason in terms of shift estimation DPU2DMA works worse. Probably because of DMA queue starvation.In
    // case of DMA2DPU synchronization DPU tasks executes almost  immediately after barrier, while in opposite case DMA
    // queue should be filled before start
    VPUX_THROW_WHEN(inverseAlgorithm, "DPU2DMA algorithm is disabled");
    using TimeType = RawProfilingRecord::TimeType;

    const auto syncPoints = findSynchronizationPoints(dmaTasks, otherTasks, pointKind);
    if (syncPoints.empty()) {
        log.warning("Cannot find synchronization points for timers shift estimation. Tasks will be aligned on zero.");
        return std::nullopt;
    }
    const auto perBarrirShiftEstimation =
            getBarrierShiftEstimations(syncPoints, frequenciesSetup, log, /*extraVerbosity=*/true);

    std::optional<TimeType> rawTimerShift;
    for (TimeType shiftEstimate : perBarrirShiftEstimation) {
        if (!rawTimerShift.has_value()) {
            rawTimerShift = shiftEstimate;
        }
        if (!inverseAlgorithm) {
            rawTimerShift = std::max(rawTimerShift.value(), shiftEstimate);
        } else {
            rawTimerShift = std::min(rawTimerShift.value(), shiftEstimate);
        }
    }
    if (inverseAlgorithm) {
        rawTimerShift = -rawTimerShift.value();
    }

    const auto timersShift = static_cast<int64_t>(rawTimerShift.value());
    return timersShift;
}

void fillTaskInfoWithParsedRawRecords(std::vector<TaskInfo>& vec, const RawProfilingRecords& rawTasks,
                                      FrequenciesSetup frequenciesSetup) {
    for (const auto& task : rawTasks) {
        vec.push_back(task->getTaskInfo(frequenciesSetup));
    }
}

bool minStartTimeTaskComparator(const TaskInfo& a, const TaskInfo& b) {
    return a.start_time_ns < b.start_time_ns;
};

std::optional<uint64_t> getEarliestTaskBegin(const std::vector<TaskInfo>& tasks) {
    if (tasks.empty()) {
        return std::nullopt;
    }

    return std::min_element(tasks.cbegin(), tasks.cend(), minStartTimeTaskComparator)->start_time_ns;
}

// Lets find the minimal offset between timers as we know for sure that DPU/SW task are started after the end of
// DMA task because they are connected via the same barrier
int64_t getTimersOffset(const std::optional<int64_t> maybeTaskTimerDiff,
                        const std::optional<uint64_t> maybeEarliestDmaNs, uint64_t earliestTaskNs) {
    if (maybeTaskTimerDiff.has_value()) {
        return maybeTaskTimerDiff.value();
    }

    // Could not calculate offset between timers(Most likely DMA profiling is disabled)
    // -> set offset based on begin time
    if (maybeEarliestDmaNs.has_value()) {
        return maybeEarliestDmaNs.value() - earliestTaskNs;
    } else {
        // FIXME: we do not need to mix unsigned and singed here
        // currenntly, we apply workaround to enable a strict check
        // for such kind of errors
        // E#65384
        return -(int64_t)earliestTaskNs;
    }
};

// Adjust all tasks to zero point (earliest DMA task)
void adjustZeroPoint(std::vector<vpux::profiling::TaskInfo>& taskInfo, int64_t timerDiff,
                     const std::optional<uint64_t> maybeFirstDma) {
    const auto firstDma = static_cast<int64_t>(maybeFirstDma.value_or(0));
    for (auto& task : taskInfo) {
        int64_t startTimeNs = task.start_time_ns - firstDma + timerDiff;
        task.start_time_ns = std::max(startTimeNs, (int64_t)0);
    }
};

RawProfilingRecords parseDmaHwTaskProfiling(
        const flatbuffers::Vector<flatbuffers::Offset<ProfilingFB::DMATask>>* dmaTaskList, const void* output,
        size_t outputLen) {
    if (dmaTaskList == nullptr) {
        return {};
    }
    VPUX_THROW_UNLESS(outputLen % sizeof(HwpDma40Data_t) == 0, "Invalid profiling data");

    const size_t totalDmaTasks = (outputLen / sizeof(HwpDma40Data_t));

    RawProfilingRecords rawRecords;
    for (const ProfilingFB::DMATask* task : *dmaTaskList) {
        VPUX_THROW_WHEN(task->isProfBegin(), "DMA HWP do not use profBegin tasks");

        unsigned recordNumber = task->dataIndex();
        VPUX_THROW_UNLESS(recordNumber < totalDmaTasks, "Can't process DMA profiling data.");
        VPUX_THROW_UNLESS(recordNumber * sizeof(HwpDma40Data_t) < outputLen, "Invalid profiling data");

        auto outputBin = reinterpret_cast<const HwpDma40Data_t*>(output);
        const auto data = outputBin[recordNumber];
        const auto record = std::make_shared<RawProfilingDMA40Record>(data, task, recordNumber);
        record->checkDataOrDie();
        rawRecords.push_back(record);
    }
    return rawRecords;
}

RawProfilingRecords parseDmaSwTaskProfiling(
        const flatbuffers::Vector<flatbuffers::Offset<ProfilingFB::DMATask>>* dmaTaskList, const void* output,
        size_t outputLen, MVCNN::TargetDevice device) {
    if (dmaTaskList == nullptr) {
        return {};
    }

    uint64_t overflowShift = 0;
    uint32_t lastTime = 0;

    size_t totalDmaTasks = 0;
    if (device != MVCNN::TargetDevice::TargetDevice_VPUX37XX) {
        VPUX_THROW_WHEN(outputLen % sizeof(DMA20Data_t) != 0, "Invalid section size");
        totalDmaTasks = outputLen / sizeof(DMA20Data_t);
    } else {
        VPUX_THROW_WHEN(outputLen % sizeof(DMA27Data_t) != 0, "Invalid section size");
        totalDmaTasks = outputLen / sizeof(DMA27Data_t);
    }

    RawProfilingRecord::BarriersSet lastProfilingRecordWaitBarriers;
    uint32_t foundDmaTasks = 0;
    RawProfilingRecords rawRecords;
    for (const ProfilingFB::DMATask* taskMetadata : *dmaTaskList) {
        if (!taskMetadata->isProfBegin()) {
            foundDmaTasks++;

            unsigned recordNumber = taskMetadata->dataIndex();
            const auto updateBarriers = RawProfilingRecord::getUpdateBarriersFromTask(taskMetadata);

            VPUX_THROW_UNLESS(recordNumber < totalDmaTasks, "Can't process DMA profiling data.");

            if (device != MVCNN::TargetDevice::TargetDevice_VPUX37XX) {
                VPUX_THROW_WHEN(recordNumber * sizeof(DMA20Data_t) >= outputLen, "Invalid profiling data");

                auto outputBin = reinterpret_cast<const DMA20Data_t*>(output);
                const auto record = outputBin[recordNumber];

                // Catch overflow and increase overflow shift for absolute start time
                if (lastTime > 0x7F000000 && record.startCycle < 0x7F000000) {
                    overflowShift += 0x100000000;
                }
                lastTime = record.startCycle;

                rawRecords.push_back(
                        std::make_shared<RawProfilingDMA20Record>(record, taskMetadata, lastProfilingRecordWaitBarriers,
                                                                  updateBarriers, overflowShift, recordNumber));
            } else {
                VPUX_THROW_WHEN(recordNumber * sizeof(DMA27Data_t) >= outputLen, "Invalid profiling data");

                auto outputBin = reinterpret_cast<const DMA27Data_t*>(output);
                const auto record = outputBin[recordNumber];
                rawRecords.push_back(std::make_shared<RawProfilingDMA27Record>(
                        record, taskMetadata, lastProfilingRecordWaitBarriers, updateBarriers, recordNumber));
            }
        } else {
            lastProfilingRecordWaitBarriers = RawProfilingRecord::getWaitBarriersFromTask(taskMetadata);
        }
    }
    VPUX_THROW_UNLESS(totalDmaTasks == foundDmaTasks, "Unexpected number of DMA tasks in profiling data: {0} != {1}",
                      totalDmaTasks, foundDmaTasks);
    return rawRecords;
}

RawProfilingRecords parseUPATaskProfiling(
        const flatbuffers::Vector<flatbuffers::Offset<ProfilingFB::SWTask>>* upaTaskList, const void* output,
        size_t outputLen) {
    if (upaTaskList == nullptr) {
        return {};
    }

    auto outputUpa = reinterpret_cast<const UpaData_t*>(output);
    VPUX_THROW_UNLESS(outputLen % sizeof(UpaData_t) == 0, "Invalid profiling data");
    const size_t totalUpaTasks = outputLen / sizeof(UpaData_t);
    size_t foundUpaTasks = 0;

    RawProfilingRecords rawRecords;
    for (const ProfilingFB::SWTask* taskMeta : *upaTaskList) {
        auto currentPos = taskMeta->dataIndex();
        VPUX_THROW_UNLESS(currentPos < outputLen / sizeof(UpaData_t), "Unexpected end of blob in UPA profiling data.");
        foundUpaTasks++;

        const auto record = std::make_shared<RawProfilingUPARecord>(outputUpa[currentPos], taskMeta, currentPos);
        record->checkDataOrDie();
        rawRecords.push_back(record);
    }
    VPUX_THROW_UNLESS(totalUpaTasks == foundUpaTasks, "Unexpected number of UPA tasks in profiling data");
    return rawRecords;
}

RawProfilingRecords parseActShaveTaskProfiling(
        const flatbuffers::Vector<flatbuffers::Offset<ProfilingFB::SWTask>>* shaveTaskList, const void* output,
        size_t outputLen) {
    if (shaveTaskList == nullptr) {
        return {};
    }

    const ActShaveData_t* outputShave = reinterpret_cast<const ActShaveData_t*>(output);
    VPUX_THROW_UNLESS(outputLen % sizeof(ActShaveData_t) == 0, "Invalid profiling data");
    const size_t numOfActShaveTasks = outputLen / sizeof(ActShaveData_t);
    size_t foundActShaveTasks = 0;

    RawProfilingRecords rawRecords;
    for (const ProfilingFB::SWTask* taskMeta : *shaveTaskList) {
        size_t currentPos =
                taskMeta->bufferOffset() + taskMeta->clusterSize() * taskMeta->clusterId() + taskMeta->dataIndex();

        VPUX_THROW_UNLESS(currentPos < numOfActShaveTasks, "Unexpected end of blob in ACT section.");
        foundActShaveTasks++;
        const auto record = std::make_shared<RawProfilingACTRecord>(outputShave[currentPos], taskMeta, currentPos);
        record->checkDataOrDie();
        rawRecords.push_back(record);
    }
    VPUX_THROW_UNLESS(foundActShaveTasks == shaveTaskList->size(), "All ActShave tasks should be profiled");
    return rawRecords;
}

size_t getDpuRecordSize(MVCNN::TargetDevice device) {
    switch (device) {
    case MVCNN::TargetDevice::TargetDevice_VPUX37XX:
        return sizeof(HwpDpu27Mode0Data_t);
    case MVCNN::TargetDevice::TargetDevice_VPUX30XX:
        return sizeof(SwDpuData_t);
    default:
        VPUX_THROW("TargetDevice {0} is not supported ", MVCNN::EnumNameTargetDevice(device));
    }
}

struct DpuMetaComparator {
    bool operator()(const ProfilingFB::DPUTask* a, const ProfilingFB::DPUTask* b) const {
        return std::make_tuple(a->bufferId(), a->clusterId(), a->taskId()) <
               std::make_tuple(b->bufferId(), b->clusterId(), b->taskId());
    }
};

RawProfilingRecords parseDPUTaskProfiling(
        const flatbuffers::Vector<flatbuffers::Offset<ProfilingFB::DPUTask>>* dpuTaskList, const void* output,
        size_t outputLen, MVCNN::TargetDevice device, bool /*ignoreSanitizationErrors*/) {
    if (dpuTaskList == nullptr) {
        return {};
    }
    const size_t recordSize = getDpuRecordSize(device);
    VPUX_THROW_UNLESS(outputLen % recordSize == 0, "Invalid profiling data");

    unsigned currentPos = 0;
    std::set<const ProfilingFB::DPUTask*, DpuMetaComparator> profInfoAggregator(dpuTaskList->begin(),
                                                                                dpuTaskList->end());
    RawProfilingRecords rawRecords;
    size_t clusterBeginning = 0;
    using TaskLocationDescriptor = std::tuple<size_t, size_t>;
    TaskLocationDescriptor bufferAndClusterDescriptor(0, 0);
    for (const ProfilingFB::DPUTask* taskMeta : profInfoAggregator) {
        VPUX_THROW_UNLESS(taskMeta->taskId() < dpuTaskList->size() + 1, "Invalid profiling data");

        const TaskLocationDescriptor newDescriptor{taskMeta->bufferId(), taskMeta->clusterId()};
        if (newDescriptor != bufferAndClusterDescriptor) {
            clusterBeginning = currentPos;
            bufferAndClusterDescriptor = newDescriptor;
        }
        for (uint32_t variantId = 0; variantId < taskMeta->maxVariants(); variantId++) {
            std::shared_ptr<RawProfilingDPURecord> record;
            if (variantId < taskMeta->numVariants()) {
                const auto inClusterIndex = currentPos - clusterBeginning;
                VPUX_THROW_WHEN(currentPos >= outputLen / recordSize, "Profiling index is out of range");
                if (device == MVCNN::TargetDevice::TargetDevice_VPUX37XX) {
                    const HwpDpu27Mode0Data_t dpuTimings =
                            reinterpret_cast<const HwpDpu27Mode0Data_t*>(output)[currentPos];
                    record = std::make_shared<RawProfilingDPUHW27Record>(dpuTimings, taskMeta, variantId, currentPos,
                                                                         inClusterIndex);
                } else {
                    const SwDpuData_t dpuTimings = reinterpret_cast<const SwDpuData_t*>(output)[currentPos];
                    record = std::make_shared<RawProfilingDPUSWRecord>(dpuTimings, taskMeta, variantId, currentPos,
                                                                       inClusterIndex);
                }
                record->checkDataOrDie();
                rawRecords.push_back(record);
            }
            // continue increment of currentPos to walk over non-used data
            ++currentPos;
        }
    }
    return rawRecords;
}

std::vector<std::pair<WorkpointConfiguration_t, size_t>> getWorkpointData(const void* output, size_t outputLen,
                                                                          size_t offset) {
    const auto NUM_WORKPOINTS = 2;
    VPUX_THROW_UNLESS(outputLen == vpux::WORKPOINT_BUFFER_SIZE, "Unexpected workpoint size: {0}", outputLen);
    const auto* workpointsPtr = reinterpret_cast<const WorkpointConfiguration_t*>(output);

    std::vector<std::pair<WorkpointConfiguration_t, size_t>> workpoints;
    for (size_t i = 0; i < NUM_WORKPOINTS; ++i) {
        workpoints.emplace_back(workpointsPtr[i], offset);
        offset += sizeof(WorkpointConfiguration_t);
    }

    return workpoints;
}

RawProfilingData parseProfilingTaskLists(const RawDataLayout& sections, MVCNN::TargetDevice device,
                                         const uint8_t* profData, const ProfilingFB::ProfilingMeta* profilingSchema,
                                         bool ignoreSanitizationErrors) {
    RawProfilingData rawProfData;

    for (const auto& section : sections) {
        const auto offset = section.second.first;
        const auto length = section.second.second;

        switch (section.first) {
        case ExecutorType::DMA_SW: {
            rawProfData.dmaTasks =
                    parseDmaSwTaskProfiling(profilingSchema->dmaTasks(), profData + offset, length, device);
            rawProfData.parseOrder.emplace_back(ExecutorType::DMA_SW, offset);
            break;
        }
        case ExecutorType::DMA_HW: {
            rawProfData.dmaTasks = parseDmaHwTaskProfiling(profilingSchema->dmaTasks(), profData + offset, length);
            rawProfData.parseOrder.emplace_back(ExecutorType::DMA_HW, offset);
            break;
        }
        case ExecutorType::UPA: {
            rawProfData.swTasks = parseUPATaskProfiling(profilingSchema->swTasks(), profData + offset, length);
            rawProfData.parseOrder.emplace_back(ExecutorType::UPA, offset);
            break;
        }
        case ExecutorType::ACTSHAVE: {
            rawProfData.swTasks = parseActShaveTaskProfiling(profilingSchema->swTasks(), profData + offset, length);
            rawProfData.parseOrder.emplace_back(ExecutorType::ACTSHAVE, offset);
            break;
        }
        case ExecutorType::DPU: {
            rawProfData.dpuTasks = parseDPUTaskProfiling(profilingSchema->dpuTasks(), profData + offset, length, device,
                                                         ignoreSanitizationErrors);
            rawProfData.parseOrder.emplace_back(ExecutorType::DPU, offset);
            break;
        }
        case ExecutorType::WORKPOINT: {
            const auto isWorkpointAccessible = device == MVCNN::TargetDevice::TargetDevice_VPUX37XX;
            if (isWorkpointAccessible && length != 0) {
                rawProfData.workpoints = getWorkpointData(profData + offset, length, offset);
            }
            break;
        }
        default:
            VPUX_THROW("Invalid profiling executor.");
        }
    }
    return rawProfData;
}

}  // namespace

FrequenciesSetup FrequenciesSetup::get30XXSetup(double nceFreq) {
    FrequenciesSetup freq;
    freq.profClk = nceFreq;
    freq.dmaBandwidth = Dma20Bandwidth;
    return freq;
}

FrequenciesSetup FrequenciesSetup::get37XXSetup(uint16_t pllMult) {
    if (pllMult < 10 || pllMult > 42) {
        vpux::Logger::global().warning("PLL multiplier '{0}' is out of [10; 42] range. MAX freq. setup will be used.",
                                       pllMult);
        return getFreqSetup37XXHelper(975.0, 1300.0, PROF_CLK_37XX_DEFAULT_VALUE_MHZ);
    }
    const double VPU_TO_PLL_RATIO = 25.0;
    const double DPU_TO_VPU_RATIO = 4.0 / 3.0;

    const double vpuFreq = pllMult * VPU_TO_PLL_RATIO;
    const double dpuFreq = vpuFreq * DPU_TO_VPU_RATIO;
    return getFreqSetup37XXHelper(vpuFreq, dpuFreq, PROF_CLK_37XX_DEFAULT_VALUE_MHZ);
}

double getNceFreq(const MVCNN::GraphFile* graphFile) {
    double frcSpeedMhz = 0;
    auto processor_frequencies = graphFile->header()->resources()->processor_frequencies();
    VPUX_THROW_UNLESS(processor_frequencies, "Blob contains no processor_frequencies");
    for (auto frequency : *processor_frequencies) {
        if (frequency->item() == MVCNN::PhysicalProcessor_NCE_Cluster) {
            frcSpeedMhz = frequency->number();
            break;
        }
    }

    if (!frcSpeedMhz) {
        switch (graphFile->header()->device()) {
        case MVCNN::TargetDevice::TargetDevice_VPUX30XX:
            switch (graphFile->header()->device_revision()) {
            case MVCNN::TargetDeviceRevision::TargetDeviceRevision_A0:
                frcSpeedMhz = 500;
                break;
            case MVCNN::TargetDeviceRevision::TargetDeviceRevision_B0:
                frcSpeedMhz = 700;
                break;
            default:
                VPUX_THROW("TargetDeviceRevision {0} is not supported",
                           EnumNameTargetDeviceRevision(graphFile->header()->device_revision()));
            }
            break;
        default:
            VPUX_THROW("TargetDevice {0} is not supported ", EnumNameTargetDevice(graphFile->header()->device()));
        }
    }

    return frcSpeedMhz;
}

RawDataLayout getRawDataLayoutFB(const ProfilingFB::ProfilingBuffer* profBuffer, size_t actualBufferSize) {
    VPUX_THROW_UNLESS(profBuffer != nullptr, "Profiling buffer data must be not empty");

    const uint32_t profSize = profBuffer->size();
    VPUX_THROW_WHEN(uint32_t(actualBufferSize) != profSize,
                    "The profiling data size does not match the expected size. Expected {0}, but got {1}", profSize,
                    actualBufferSize);

    uint32_t prevSectionEnd = 0;
    RawDataLayout sections;
    for (const auto& section : *profBuffer->sections()) {
        const auto sectionBegin = section->offset();
        const auto sectionSize = section->size();
        const auto sectionEnd = sectionBegin + sectionSize;

        VPUX_THROW_UNLESS((sectionBegin < profSize && sectionEnd <= profSize),
                          "Section [{0};{1}] exceeds profiling buffer size({2}b)", sectionBegin, sectionEnd, profSize);
        VPUX_THROW_WHEN(sectionBegin < prevSectionEnd, "Section(type {0}) overlaps with previous section",
                        section->type());

        const auto execType = static_cast<ExecutorType>(section->type());
        sections[execType] = {sectionBegin, sectionSize};
        prevSectionEnd = sectionEnd;
    }
    return sections;
}

TokenizedTaskName vpux::profiling::tokenizeTaskName(const std::string& taskName) {
    auto nameSepPos = taskName.rfind(vpux::LOCATION_ORIGIN_SEPARATOR);
    VPUX_THROW_WHEN(nameSepPos == std::string::npos, "Malformed task name: '{0}'", taskName);
    auto layerName = taskName.substr(0, nameSepPos);
    auto afterNameSep = taskName.substr(nameSepPos + 1);
    std::vector<std::string> parts = splitBySeparator(afterNameSep, vpux::LOCATION_SEPARATOR);
    return {std::move(layerName), std::move(parts)};
}

ParsedTaskName vpux::profiling::deserializeTaskName(const std::string& fullTaskName) {
    const auto LOC_METADATA_SEPARATOR = '_';  // conventional separator used for attaching metadata to MLIR Locations

    auto tokenized = tokenizeTaskName(fullTaskName);
    std::string layerType = "<unknown>";
    std::string& layerName = tokenized.layerName;

    for (const auto& token : tokenized.tokens) {
        VPUX_THROW_WHEN(token.empty(), "Empty task name token");

        auto parts = splitBySeparator(token, LOC_METADATA_SEPARATOR);
        auto partsNum = parts.size();

        if (partsNum == 2 && parts[0] == vpux::LOCATION_LAYER_TYPE_PREFIX) {
            layerType = parts[1];
        }
    }

    return {std::move(layerName), std::move(layerType)};
}

std::string getLayerName(const std::string& taskName) {
    return taskName.substr(0, taskName.rfind(vpux::LOCATION_ORIGIN_SEPARATOR));
}

RawProfilingRecords RawProfilingData::getTaskOfType(ExecutorType type) const {
    switch (type) {
    case ExecutorType::DMA_HW:
    case ExecutorType::DMA_SW:
        return dmaTasks;
    case ExecutorType::DPU:
        return dpuTasks;
    case ExecutorType::UPA:
    case ExecutorType::ACTSHAVE:
        return swTasks;
    default:
        VPUX_THROW("Unsupported executor type");
    }
}

uint16_t getPllValueChecked(const WorkpointRecords& workpoints, vpux::Logger& log) {
    VPUX_THROW_WHEN(workpoints.empty(), "Expected workpoint data");
    // PLL value from begin of inference
    const auto pllMultFirst = workpoints.front().first.pllMultiplier;
    // PLL value from end of inference
    const auto pllMultLast = workpoints.back().first.pllMultiplier;
    if (pllMultFirst != pllMultLast) {
        log.warning("Frequency changed during the inference: {0} != {1}", pllMultFirst, pllMultLast);
    }
    return pllMultFirst;
}

RawData vpux::profiling::getRawProfilingTasks(const uint8_t* blobData, size_t blobSize, const uint8_t* profData,
                                              size_t profSize, bool ignoreSanitizationErrors) {
    if ((nullptr == blobData) || (nullptr == profData)) {
        VPUX_THROW("Empty input data");
    }

    std::optional<double> maybe30XXFreq;
    if (!vpux::profiling::isElfBinary(blobData, blobSize)) {
        const auto graphFile = vpux::profiling::getGraphFileVerified(blobData, blobSize);
        if (graphFile->header()->device() == MVCNN::TargetDevice::TargetDevice_VPUX30XX) {
            maybe30XXFreq = getNceFreq(graphFile);
        }
    }

    const auto log = vpux::Logger::global();
    const auto profilingDataSchema = vpux::profiling::getProfilingSectionMeta(blobData, blobSize);
    auto device = (MVCNN::TargetDevice)profilingDataSchema->platform()->device();
    VPUX_THROW_WHEN(device == MVCNN::TargetDevice::TargetDevice_NONE, "Unknown device");
    log.trace("Using target device {0}", MVCNN::EnumNameTargetDevice(device));

    const auto profilingBufferMeta = profilingDataSchema->profilingBuffer();
    const auto sections = getRawDataLayoutFB(profilingBufferMeta, profSize);

    RawProfilingData rawProfData =
            parseProfilingTaskLists(sections, device, profData, profilingDataSchema, ignoreSanitizationErrors);

    return {sections, std::move(rawProfData), device, maybe30XXFreq};
}

RawProfilingRecords makeFakeDpuInvariants(const RawProfilingRecords& variants) {
    RawProfilingRecords invariants;

    // Grouping of variants into one invariant
    std::multimap<std::pair<std::string, size_t>, RawProfilingRecordPtr> groupedClustersInfo;
    for (const auto& task : variants) {
        const auto clusteredTask = std::dynamic_pointer_cast<RawProfilingDPURecord>(task);
        VPUX_THROW_WHEN(clusteredTask == nullptr, "Expected cluster task");
        const auto clusterId = clusteredTask->getClusterId();
        const auto key = std::make_pair(task->getOriginalName(), clusterId);
        groupedClustersInfo.insert(std::make_pair(key, task));
    }

    auto it = groupedClustersInfo.cbegin();
    while (it != groupedClustersInfo.cend()) {
        RawProfilingRecords variants;
        const auto groupingKey = it->first;
        std::string name = groupingKey.first;

        while (it != groupedClustersInfo.cend() && it->first == groupingKey) {
            variants.push_back(it->second);
            ++it;
        }
        invariants.push_back(std::make_shared<ArrayRecord>(name, variants));
    }

    return invariants;
}

FrequenciesSetup getFrequencySetup(const MVCNN::TargetDevice device, const WorkpointRecords& workpoints,
                                   const std::optional<double>& maybe30XXNceFreq, bool highFreqPerfClk, bool fpga,
                                   vpux::Logger& log) {
    const bool isDeviceSupportsHighPerfClk = false;
    if (!isDeviceSupportsHighPerfClk && highFreqPerfClk) {
        log.warning("Requested perf_clk high frequency value is not supported on this device. Default value for the "
                    "device will be used for frequency setup.");
    }
    const bool isPllCaptureSupported = device == MVCNN::TargetDevice::TargetDevice_VPUX37XX;
    FrequenciesSetup frequenciesSetup;

    if (isPllCaptureSupported) {
        uint16_t pllMult = 0;
        if (workpoints.size()) {
            pllMult = getPllValueChecked(workpoints, log);
        } else {
            log.warning("No frequency data");
        }
        log.trace("Got PLL value '{0}'", pllMult);

        frequenciesSetup = fpga ? getFpgaFreqSetup(device) : getFreqSetupFromPll(device, pllMult, highFreqPerfClk);
    } else if (device == MVCNN::TargetDevice::TargetDevice_VPUX30XX) {
        VPUX_THROW_UNLESS(maybe30XXNceFreq.has_value(), "No frequency data");
        frequenciesSetup.profClk = maybe30XXNceFreq.value();
        frequenciesSetup.dmaBandwidth = Dma20Bandwidth;
    } else {
        VPUX_THROW("TargetDevice {0} is not supported ", MVCNN::EnumNameTargetDevice(device));
    }
    log.trace("Frequency setup is profClk={0}MHz, vpuClk={1}MHz, dpuClk={2}MHz", frequenciesSetup.profClk,
              frequenciesSetup.vpuClk, frequenciesSetup.dpuClk);

    return frequenciesSetup;
}

// At parse time we don't know frequency for some platforms, so data is collected in cycles format. We need
// to determine frequency to convert from cycles to nanoseconds
std::vector<TaskInfo> convertRawTasksToTaskInfo(const RawData& profData, bool fpga, VerbosityLevel verbosity,
                                                bool highFreqPerfClk) {
    auto log = vpux::Logger::global();
    const auto rawTasks = profData.rawRecords;
    const auto device = profData.device;

    FrequenciesSetup frequenciesSetup =
            getFrequencySetup(device, rawTasks.workpoints, profData.maybe30XXNceFreq, highFreqPerfClk, fpga, log);

    for (const auto& taskList : {rawTasks.dmaTasks, rawTasks.dpuTasks, rawTasks.swTasks}) {
        for (const auto& task : taskList) {
            task->sanitize(log, frequenciesSetup);
        }
    }

    std::vector<TaskInfo> dmaTaskInfo;
    std::vector<TaskInfo> swTaskInfo;
    std::vector<TaskInfo> dpuTaskInfo;

    fillTaskInfoWithParsedRawRecords(dmaTaskInfo, rawTasks.dmaTasks, frequenciesSetup);

    fillTaskInfoWithParsedRawRecords(swTaskInfo, rawTasks.swTasks, frequenciesSetup);

    RawProfilingRecords dpuInvariantTasks = makeFakeDpuInvariants(rawTasks.dpuTasks);
    fillTaskInfoWithParsedRawRecords(dpuTaskInfo, dpuInvariantTasks, frequenciesSetup);
    if (verbosity >= VerbosityLevel::MEDIUM) {
        fillTaskInfoWithParsedRawRecords(dpuTaskInfo, rawTasks.dpuTasks, frequenciesSetup);
    }

    const auto earliestDpuNs = getEarliestTaskBegin(dpuTaskInfo);
    const auto earliestDmaNs = getEarliestTaskBegin(dmaTaskInfo);
    const auto earliestSwNs = getEarliestTaskBegin(swTaskInfo);

    log.trace("Earliest DMA: {0}", earliestDmaNs);
    log.trace("Earliest DPU: {0}", earliestDpuNs);
    log.trace("Earliest SW : {0}", earliestSwNs);

    adjustZeroPoint(dmaTaskInfo, 0, earliestDmaNs);

    if (!dpuTaskInfo.empty()) {
        int64_t dma2dpuOffset = 0;
        if (!frequenciesSetup.hasSharedDmaDpuCounter) {
            const auto timersShift = getDMA2OtherTimersShift(rawTasks.dmaTasks, rawTasks.dpuTasks, frequenciesSetup,
                                                             SynchronizationPointKind::STRICT_DMA_TO_DPU, log);
            log.trace("Timers DMA2DPU difference: {0}", timersShift);
            dma2dpuOffset = getTimersOffset(timersShift, earliestDmaNs, earliestDpuNs.value());
        } else {
            // If DMA profiling enabled difference is 0, otherwise setting to earliest task without call to
            // getTimersOffset to avoid counting twice
            dma2dpuOffset = earliestDmaNs.has_value() ? 0 : -static_cast<int64_t>(earliestDpuNs.value());
        }
        adjustZeroPoint(dpuTaskInfo, dma2dpuOffset, earliestDmaNs);
        // Order DPU tasks by time to make tests more stable
        std::sort(dpuTaskInfo.begin(), dpuTaskInfo.end(), profilingTaskStartTimeComparator<TaskInfo>);
    }

    if (!swTaskInfo.empty()) {
        int64_t dma2SwOffset = 0;
        if (!frequenciesSetup.hasSharedDmaSwCounter) {
            const auto dma2UpaTimerShift = getDMA2OtherTimersShift(
                    rawTasks.dmaTasks, rawTasks.swTasks, frequenciesSetup, SynchronizationPointKind::DMA_TO_UPA, log);
            log.trace("Timers DMA2UPA difference: {0}", dma2UpaTimerShift);
            dma2SwOffset = getTimersOffset(dma2UpaTimerShift, earliestDmaNs, earliestSwNs.value());
        } else {
            // If DMA profiling enabled difference is 0, otherwise setting to earliest task without call to
            // getTimersOffset to avoid counting twice
            dma2SwOffset = earliestDmaNs.has_value() ? 0 : -static_cast<int64_t>(earliestSwNs.value());
        }
        adjustZeroPoint(swTaskInfo, dma2SwOffset, earliestDmaNs);
    }

    std::vector<TaskInfo> allTaskInfo;
    allTaskInfo.reserve(dpuTaskInfo.size() + dmaTaskInfo.size() + swTaskInfo.size());
    allTaskInfo.insert(allTaskInfo.end(), dpuTaskInfo.begin(), dpuTaskInfo.end());
    allTaskInfo.insert(allTaskInfo.end(), dmaTaskInfo.begin(), dmaTaskInfo.end());
    allTaskInfo.insert(allTaskInfo.end(), swTaskInfo.begin(), swTaskInfo.end());

    return allTaskInfo;
}

std::vector<TaskInfo> vpux::profiling::getTaskInfo(const uint8_t* blobData, size_t blobSize, const uint8_t* profData,
                                                   size_t profSize, VerbosityLevel verbosity, bool fpga,
                                                   bool ignoreSanitizationErrors) try {
    const auto rawProfData = getRawProfilingTasks(blobData, blobSize, profData, profSize, ignoreSanitizationErrors);
    return convertRawTasksToTaskInfo(rawProfData, fpga, verbosity, false);
} catch (const std::exception& ex) {
    VPUX_THROW("Profiling post-processing failed. {0}", ex.what());
}

std::vector<LayerInfo> vpux::profiling::getLayerInfo(const uint8_t* blobData, size_t blobSize, const uint8_t* profData,
                                                     size_t profSize, bool fpga, bool ignoreSanitizationErrors) {
    std::vector<TaskInfo> taskInfo =
            getTaskInfo(blobData, blobSize, profData, profSize, VerbosityLevel::LOW, fpga, ignoreSanitizationErrors);
    return getLayerInfo(taskInfo);
}

std::vector<LayerInfo> vpux::profiling::getLayerInfo(const std::vector<TaskInfo>& taskInfo) {
    std::vector<LayerInfo> layerInfo;
    for (const auto& task : taskInfo) {
        LayerInfo* layer;
        std::string taskName(task.name);
        if (!getVariantFromName(taskName).empty()) {
            // Skipping high verbose tasks with variant info
            continue;
        }

        std::string layerName = getLayerName(taskName);
        auto result = std::find_if(begin(layerInfo), end(layerInfo), [&](const LayerInfo& item) {
            return layerName == item.name;
        });
        if (result == end(layerInfo)) {
            layer = &layerInfo.emplace_back();
            layer->status = LayerInfo::layer_status_t::EXECUTED;
            layer->start_time_ns = task.start_time_ns;
            layer->duration_ns = 0;

            const auto nameLen = layerName.copy(layer->name, sizeof(layer->name) - 1);
            layer->name[nameLen] = 0;

            const std::string layerTypeStr(task.layer_type);
            const auto typeLen = layerTypeStr.copy(layer->layer_type, sizeof(layer->layer_type) - 1);
            layer->layer_type[typeLen] = 0;
        } else {
            layer = &(*result);
        }
        if (task.start_time_ns < layer->start_time_ns) {
            layer->duration_ns += layer->start_time_ns - task.start_time_ns;
            layer->start_time_ns = task.start_time_ns;
        }
        auto duration = (int64_t)task.start_time_ns + task.duration_ns - layer->start_time_ns;
        if (duration > layer->duration_ns) {
            layer->duration_ns = duration;
        }

        if (task.exec_type == TaskInfo::ExecType::DPU) {
            layer->dpu_ns += task.duration_ns;
        } else if (task.exec_type == TaskInfo::ExecType::SW || task.exec_type == TaskInfo::ExecType::UPA) {
            layer->sw_ns += task.duration_ns;
        } else if (task.exec_type == TaskInfo::ExecType::DMA) {
            layer->dma_ns += task.duration_ns;
        }
    }

    return layerInfo;
}

std::string vpux::profiling::getTaskNameSuffixes(const std::string& name) {
    const auto startPos = name.rfind(LOCATION_ORIGIN_SEPARATOR);
    if (startPos == std::string::npos) {
        return "";
    }
    return name.substr(startPos + 1);
}

std::string vpux::profiling::getClusterFromName(const std::string& name) {
    return getValueFromStructuredTaskName(name, CLUSTER_LEVEL_PROFILING_SUFFIX);
}

std::string vpux::profiling::getVariantFromName(const std::string& name) {
    return getValueFromStructuredTaskName(name, VARIANT_LEVEL_PROFILING_SUFFIX);
}

std::string vpux::profiling::getValueFromStructuredTaskName(const std::string& name, std::string key) {
    auto taskNameSuffixes = getTaskNameSuffixes(name);
    auto suffixes = splitBySeparator(taskNameSuffixes, LOCATION_SEPARATOR);

    for (auto& suffix : suffixes) {
        auto parts = splitBySeparator(suffix, '_');
        if (parts.size() == 2) {
            auto extractedKey = parts[0];
            auto extractedValue = parts[1];
            if (extractedKey == key) {
                return extractedValue;
            }
        }
    }
    return "";
}
