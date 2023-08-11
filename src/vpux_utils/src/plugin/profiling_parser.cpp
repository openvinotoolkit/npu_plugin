//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/plugin/profiling_parser.hpp"
#include "vpux/utils/IE/prefix.hpp"
#include "vpux/utils/IE/profiling.hpp"
#include "vpux/utils/core/profiling.hpp"

#include <vpux_elf/accessor.hpp>
#include <vpux_elf/reader.hpp>
#include <vpux_elf/utils/utils.hpp>

#include <map>
#include <sstream>
#include <string>

using namespace vpux::profiling;

// Main synchronization primitive. First contain tasks, that update barrier, Second - that wait for barrier
using SynchronizationPoint = std::pair<RawProfilingRecords, RawProfilingRecords>;
using SynchronizationPointsContainer = std::vector<SynchronizationPoint>;
using ClusterTaskArray = std::vector<std::shared_ptr<ArrayRecord>>;

enum class SynchronizationPointKind { DMA_TO_DPU, DPU_TO_DMA, DMA_TO_UPA, STRICT_DMA_TO_DPU };

// These two declarations serve to make RawProfilingDMARecord class usable in other compilation units, i.e. unit tests
template class vpux::profiling::RawProfilingDMARecord<DMA20Data_t>;
template class vpux::profiling::RawProfilingDMARecord<DMA27Data_t>;

namespace {

constexpr double PROF_CLK_VALUE_MHZ = 38.4;

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

constexpr auto MAX_37XX_FREQ_SETUP = getFreqSetup37XXHelper(975.0, 1300.0, PROF_CLK_VALUE_MHZ);

FrequenciesSetup getFpgaFreqSetup(MVCNN::TargetDevice device) {
    VPUX_THROW("TargetDevice {0} is not supported ", MVCNN::EnumNameTargetDevice(device));
}

FrequenciesSetup getFreqSetupFromPll(MVCNN::TargetDevice device, uint16_t pll) {
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

std::string to_string(const SynchronizationPoint& syncPoint) {
    const auto getType = [](const auto& vec) -> std::string {
        switch (vec.front()->getExecutorType()) {
        case ExecutorType::DMA_SW:
            return "DMA_SW";
        case ExecutorType::DMA_HW:
            return "DMA_HW";
        case ExecutorType::DPU:
            return "DPU";
        case ExecutorType::UPA:
            return "UPA";
        case ExecutorType::ACTSHAVE:
            return "ACTSHAVE";
        case ExecutorType::WORKPOINT:
            return "WORKPOINT";
        case ExecutorType::NONE:
            return "NONE";
        };
        return "NONE";
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

using BarriersSet = RawProfilingRecord::BarriersSet;

template <class TaskType>
BarriersSet getBarriersFromTask(const TaskType* task, bool waitBarriers) {
    if (task == nullptr) {
        return {};
    }
    const auto barrierList = waitBarriers ? task->waitBarriers() : task->updateBarriers();
    if (barrierList->size() == 0) {
        return {};
    }
    return BarriersSet(barrierList->cbegin(), barrierList->cend());
}

template <>
BarriersSet getBarriersFromTask(const MVCNN::Task* task, bool waitBarriers) {
    if (task == nullptr) {
        return {};
    }
    if (task->associated_barriers() == nullptr) {
        return {};
    }
    const auto associatedBarriers = task->associated_barriers();
    const auto barrierList = waitBarriers ? associatedBarriers->wait_barriers() : associatedBarriers->update_barriers();
    if (auto list = barrierList) {
        return BarriersSet(list->cbegin(), list->cend());
    }
    return {};
}

void checkSameBarriers(const RawProfilingRecordPtr& first, const RawProfilingRecordPtr& second,
                       bool checkWaitBarriers) {
    auto firstBarriers = checkWaitBarriers ? first->getWaitBarriers() : first->getUpdateBarriers();
    auto secondBarriers = checkWaitBarriers ? second->getWaitBarriers() : second->getUpdateBarriers();
    const auto intersection = RawProfilingRecord::getBarriersIntersection(firstBarriers, secondBarriers);
    bool barriersAreSame = intersection.size() == firstBarriers.size();
    if (!barriersAreSame) {
        const auto barrierKind = checkWaitBarriers ? "wait" : "update";
        VPUX_THROW("Tasks must have same {0} barriers, but {1} != {2}", barrierKind,
                   convertIterableToString(firstBarriers), convertIterableToString(secondBarriers));
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
llvm::Optional<int64_t> getDMA2OtherTimersShift(const RawProfilingRecords& dmaTasks,
                                                const RawProfilingRecords& otherTasks,
                                                FrequenciesSetup frequenciesSetup, SynchronizationPointKind pointKind,
                                                vpux::Logger& log) {
    const auto inverseAlgorithm = pointKind == SynchronizationPointKind::DPU_TO_DMA;
    // For some reason in terms of shift estimation DPU2DMA works worse. Probably because of DMA queue starvation.In
    // case of DMA2DPU synchronization DPU tasks executes almost  immediately after barrier, while in opposite case DMA
    // queue should be filled before start
    VPUX_THROW_WHEN(inverseAlgorithm, "DPU2DMA algorithm is disabled");
    using TimeType = RawProfilingRecord::TimeType;

    const auto syncPoints = findSynchronizationPoints(dmaTasks, otherTasks, pointKind);
    if (syncPoints.empty()) {
        log.warning("Cannot find synchronization points for timers shift estimation. Tasks will be aligned on zero.");
        return llvm::None;
    }
    const auto perBarrirShiftEstimation =
            getBarrierShiftEstimations(syncPoints, frequenciesSetup, log, /*extraVerbosity=*/true);

    llvm::Optional<TimeType> rawTimerShift;
    for (TimeType shiftEstimate : perBarrirShiftEstimation) {
        if (!rawTimerShift.hasValue()) {
            rawTimerShift = shiftEstimate;
        }
        if (!inverseAlgorithm) {
            rawTimerShift = std::max(rawTimerShift.getValue(), shiftEstimate);
        } else {
            rawTimerShift = std::min(rawTimerShift.getValue(), shiftEstimate);
        }
    }
    if (inverseAlgorithm) {
        rawTimerShift = -rawTimerShift.getValue();
    }

    const auto timersShift = static_cast<int64_t>(rawTimerShift.getValue());
    return timersShift;
}

template <class InnerType>
void fillTaskInfoWithParsedRawRecords(std::vector<TaskInfo>& vec,
                                      const std::vector<std::shared_ptr<InnerType>>& rawTasks,
                                      FrequenciesSetup frequenciesSetup) {
    for (const auto& task : rawTasks) {
        vec.push_back(task->getTaskInfo(frequenciesSetup));
    }
}

bool minStartTimeTaskComparator(const TaskInfo& a, const TaskInfo& b) {
    return a.start_time_ns < b.start_time_ns;
};

llvm::Optional<uint64_t> getEarliestTaskBegin(const std::vector<TaskInfo>& tasks) {
    if (tasks.empty()) {
        return llvm::None;
    }

    return std::min_element(tasks.cbegin(), tasks.cend(), minStartTimeTaskComparator)->start_time_ns;
}

// Lets find the minimal offset between timers as we know for sure that DPU/SW task are started after the end of
// DMA task because they are connected via the same barrier
int64_t getTimersOffset(const llvm::Optional<int64_t> maybeTaskTimerDiff,
                        const llvm::Optional<uint64_t> maybeEarliestDmaNs, uint64_t earliestTaskNs) {
    if (maybeTaskTimerDiff.hasValue()) {
        return maybeTaskTimerDiff.getValue();
    }

    // Could not calculate offset between timers(Most likely DMA profiling is disabled)
    // -> set offset based on begin time
    if (maybeEarliestDmaNs.hasValue()) {
        return maybeEarliestDmaNs.getValue() - earliestTaskNs;
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
                     const llvm::Optional<uint64_t> maybeFirstDma) {
    const auto firstDma = static_cast<int64_t>(maybeFirstDma.value_or(0));
    for (auto& task : taskInfo) {
        int64_t startTimeNs = task.start_time_ns - firstDma + timerDiff;
        task.start_time_ns = std::max(startTimeNs, (int64_t)0);
    }
};

struct TaskMetadataBase {
    template <class TaskType>
    TaskMetadataBase(const TaskType* task) {
        waitBarriers = getBarriersFromTask(task, /*waitBarriers=*/true);
        updateBarriers = getBarriersFromTask(task, /*waitBarriers=*/false);
        name = task->name()->str();
    }

    BarriersSet waitBarriers;
    BarriersSet updateBarriers;
    std::string name;
};

struct DmaTaskMetadata : public TaskMetadataBase {
    DmaTaskMetadata(const MVCNN::Task* task): TaskMetadataBase(task) {
        const auto asNnDma = task->task_as_NNDMATask();
        isFromRegister = asNnDma->src()->locale() == MVCNN::MemoryLocation_AbsoluteAddr;
        dmaHwpId = asNnDma->dma_hwp_id();
    }

    DmaTaskMetadata(const ProfilingFB::DMATask* task): TaskMetadataBase(task) {
        isFromRegister = task->sourceLocale()->str() == "Register";
        dmaHwpId = 0;
    }

    bool isFromRegister;
    uint32_t dmaHwpId;
};

template <class TaskType>
RawProfilingRecords parseDmaHwTaskProfiling(const MVCNN::GraphFile* graphFile,
                                            const flatbuffers::Vector<flatbuffers::Offset<TaskType>>* dmaTaskList,
                                            const void* output, size_t outputLen) {
    if (dmaTaskList == nullptr) {
        return {};
    }

    BarriersSet lastProfilingRecordWaitBarriers;
    uint32_t foundDmaTasks = 0;
    RawProfilingRecords rawRecords;
    for (unsigned dmaTaskListId = 0; dmaTaskListId < (*dmaTaskList).size(); dmaTaskListId++) {
        auto task = (*dmaTaskList)[dmaTaskListId];
        const auto taskMetadata = DmaTaskMetadata(task);
        if (taskMetadata.dmaHwpId == 0) {
            continue;
        }
        auto taskName = taskMetadata.name;

        if (!RawProfilingDMA40Record::isTaskBegin(taskName)) {
            const auto dmaMeta = RawProfilingDMA40Record::parseTaskName(taskName);
            foundDmaTasks++;
            unsigned recordNumber = dmaMeta.prof.curDmaId;

            const auto updateBarriers = taskMetadata.updateBarriers;

            auto outputBin = reinterpret_cast<const HwpDma40Data_t*>(output);
            const auto record = outputBin[recordNumber];
            rawRecords.push_back(std::make_shared<RawProfilingDMA40Record>(
                    record, dmaMeta.meta.taskName, dmaMeta.meta.layerName, dmaMeta.meta.layerType,
                    lastProfilingRecordWaitBarriers, updateBarriers, recordNumber));
            const auto asThrowableTask = std::dynamic_pointer_cast<ThrowableAssertMixin>(rawRecords.back());
            if (asThrowableTask != nullptr) {
                asThrowableTask->checkDataOrDie();
            } else {
                VPUX_THROW("Wrong type of HWPDMA record");
            }

        } else {
            lastProfilingRecordWaitBarriers = taskMetadata.waitBarriers;
        }
    }

    size_t totalDmaTasks = outputLen / sizeof(HwpDma40Data_t);
    if (auto* hwpBase = graphFile->header()->dma_hwp_base()) {
        // In case if hwpBase located in DDR, first task won't contain profiling data, so decreasing totalDmaTasks
        if (hwpBase->locale() == MVCNN::MemoryLocation_ProfilingOutput) {
            --totalDmaTasks;
        }
    }
    VPUX_THROW_UNLESS(totalDmaTasks == foundDmaTasks, "Unexpected number of DMA tasks in profiling data: {0} != {1}",
                      totalDmaTasks, foundDmaTasks);
    return rawRecords;
}

template <class TaskType>
RawProfilingRecords parseDmaSwTaskProfiling(const flatbuffers::Vector<flatbuffers::Offset<TaskType>>* dmaTaskList,
                                            const void* output, size_t outputLen, MVCNN::TargetDevice device) {
    if (dmaTaskList == nullptr) {
        return {};
    }

    uint64_t overflowShift = 0;
    uint32_t lastTime = 0;

    size_t totalDmaTasks = 0;
    if (device != MVCNN::TargetDevice::TargetDevice_VPUX37XX) {
        totalDmaTasks = outputLen / sizeof(DMA20Data_t);
    } else {
        totalDmaTasks = outputLen / sizeof(DMA27Data_t);
    }

    BarriersSet lastProfilingRecordWaitBarriers;
    uint32_t foundDmaTasks = 0;
    unsigned lastBeginTaskId = 0;
    RawProfilingRecords rawRecords;
    for (unsigned dmaTaskListId = 0; dmaTaskListId < (*dmaTaskList).size(); dmaTaskListId++) {
        auto task = (*dmaTaskList)[dmaTaskListId];
        const auto taskMetadata = DmaTaskMetadata(task);
        if (taskMetadata.isFromRegister) {
            auto taskName = taskMetadata.name;
            if (RawProfilingDMA27Record::isTaskWorkpointRead(taskName)) {
                continue;
            }

            if (!RawProfilingDMA27Record::isTaskBegin(taskName)) {
                const auto dmaMeta = RawProfilingDMA27Record::parseTaskName(taskName);
                foundDmaTasks++;
                unsigned recordNumber = dmaMeta.prof.curDmaId;

                const auto updateBarriers = taskMetadata.updateBarriers;

                VPUX_THROW_UNLESS(recordNumber < totalDmaTasks, "Can't process DMA profiling data.");

                unsigned tasksBetweenBeginAndEnd = dmaTaskListId - lastBeginTaskId - 1;
                VPUX_THROW_UNLESS(tasksBetweenBeginAndEnd == 1,
                                  "There's {0} DMA tasks between PROFTASKBEGIN and PROFTASKEND, expected 1",
                                  tasksBetweenBeginAndEnd);

                if (device != MVCNN::TargetDevice::TargetDevice_VPUX37XX) {
                    auto outputBin = reinterpret_cast<const DMA20Data_t*>(output);
                    const auto record = outputBin[recordNumber];

                    // Catch overflow and increase overflow shift for absolute start time
                    if (lastTime > 0x7F000000 && record.startCycle < 0x7F000000) {
                        overflowShift += 0x100000000;
                    }
                    lastTime = record.startCycle;

                    rawRecords.push_back(std::make_shared<RawProfilingDMA20Record>(
                            record, dmaMeta.meta.taskName, dmaMeta.meta.layerName, dmaMeta.meta.layerType,
                            lastProfilingRecordWaitBarriers, updateBarriers, overflowShift, recordNumber));
                } else {
                    auto outputBin = reinterpret_cast<const DMA27Data_t*>(output);
                    const auto record = outputBin[recordNumber];
                    rawRecords.push_back(std::make_shared<RawProfilingDMA27Record>(
                            record, dmaMeta.meta.taskName, dmaMeta.meta.layerName, dmaMeta.meta.layerType,
                            lastProfilingRecordWaitBarriers, updateBarriers, recordNumber));
                }
            } else {
                lastProfilingRecordWaitBarriers = taskMetadata.waitBarriers;
                lastBeginTaskId = dmaTaskListId;
            }
        }
    }
    VPUX_THROW_UNLESS(totalDmaTasks == foundDmaTasks, "Unexpected number of DMA tasks in profiling data: {0} != {1}",
                      totalDmaTasks, foundDmaTasks);
    return rawRecords;
}

template <class TaskType>
RawProfilingRecords parseUPATaskProfiling(const flatbuffers::Vector<flatbuffers::Offset<TaskType>>*, const void*,
                                          size_t) {
    VPUX_THROW("UPA profiling is supported only by GF");
}

template <>
RawProfilingRecords parseUPATaskProfiling(const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* upaTaskList,
                                          const void* output, size_t outputLen) {
    if (upaTaskList == nullptr) {
        return {};
    }

    auto outputUpa = reinterpret_cast<const UpaData_t*>(output);
    const size_t totalUpaTasks = outputLen / sizeof(UpaData_t);
    size_t foundUpaTasks = 0;

    RawProfilingRecords rawRecords;
    for (unsigned upaTaskListId = 0; upaTaskListId < upaTaskList->size(); upaTaskListId++) {
        auto task = (*upaTaskList)[upaTaskListId];
        auto taskName = task->name()->str();

        const auto upaMeta = RawProfilingUPARecord::parseTaskName(taskName);
        auto currentPos = upaMeta.prof.currentPos;
        VPUX_THROW_UNLESS(currentPos < outputLen / sizeof(UpaData_t), "Unexpected end of blob in UPA profiling data.");
        foundUpaTasks++;

        auto softLayer = task->task_as_UPALayerTask();
        std::string layerName = upaMeta.meta.layerType;
        if (softLayer != nullptr) {
            const char* typeName = EnumNameSoftwareLayerParams(softLayer->softLayerParams_type());
            if (typeName != nullptr) {
                layerName = typeName;
            }
        }

        rawRecords.push_back(std::make_shared<RawProfilingUPARecord>(
                outputUpa[currentPos], upaMeta.meta.taskName, upaMeta.meta.layerName, layerName, task, currentPos));
        std::dynamic_pointer_cast<ThrowableAssertMixin>(rawRecords.back())->checkDataOrDie();
    }
    VPUX_THROW_UNLESS(totalUpaTasks == foundUpaTasks, "Unexpected number of UPA tasks in profiling data");
    return rawRecords;
}

template <class TaskType>
RawProfilingRecords parseActShaveTaskProfiling(const flatbuffers::Vector<flatbuffers::Offset<TaskType>>* shaveTaskList,
                                               const void* output, size_t outputLen) {
    if (shaveTaskList == nullptr) {
        return {};
    }

    const ActShaveData_t* outputShave = reinterpret_cast<const ActShaveData_t*>(output);
    const size_t numOfActShaveTasks = outputLen / sizeof(ActShaveData_t);
    size_t foundActShaveTasks = 0;

    RawProfilingRecords rawRecords;
    for (const auto& task : *shaveTaskList) {
        const auto taskMetadata = TaskMetadataBase(task);
        auto taskName = taskMetadata.name;
        const auto actMeta = RawProfilingACTRecord::parseTaskName(taskName);

        size_t currentPos = actMeta.prof.getResultingDDROffset();

        VPUX_THROW_UNLESS(currentPos < numOfActShaveTasks, "Unexpected end of blob in ACT section.");
        foundActShaveTasks++;
        rawRecords.push_back(std::make_shared<RawProfilingACTRecord>(
                outputShave[currentPos], actMeta.meta.taskName, actMeta.meta.layerName, actMeta.meta.layerType,
                taskMetadata.waitBarriers, taskMetadata.updateBarriers, actMeta.prof.clusterId, actMeta.prof.tileId,
                currentPos));
        std::dynamic_pointer_cast<ThrowableAssertMixin>(rawRecords.back())->checkDataOrDie();
    }
    VPUX_THROW_UNLESS(foundActShaveTasks == shaveTaskList->size(), "All ActShave tasks should be profiled");
    return rawRecords;
}

template <class TaskType>
RawProfilingRecords parseDPUTaskProfiling(const flatbuffers::Vector<flatbuffers::Offset<TaskType>>* dpuTaskList,
                                          const void* output, size_t outputLen, MVCNN::TargetDevice device) {
    if (dpuTaskList == nullptr) {
        return {};
    }

    std::map<uint64_t, RawProfilingDPURecord::ParsedTaskNameProf> profilingMeta;
    for (unsigned dpu_taskListId = 0; dpu_taskListId < (*dpuTaskList).size(); dpu_taskListId++) {
        auto task = (*dpuTaskList)[dpu_taskListId];
        const auto taskMetadata = TaskMetadataBase(task);
        auto taskName = taskMetadata.name;
        auto meta = RawProfilingDPURecord::parseTaskName(taskName, dpu_taskListId);
        profilingMeta.emplace(meta.prof.getOrderDescriptor(), meta);
    }

    unsigned currentPos = 0;
    std::map<std::string, TaskInfo> profInfoAggregator;

    RawProfilingRecords rawRecords;
    for (const auto& iter : profilingMeta) {
        const auto metaData = iter.second;
        const auto task = (*dpuTaskList)[metaData.prof.taskId];
        const auto taskMetadata = TaskMetadataBase(task);

        for (auto variantId = 0; variantId < metaData.prof.maxVariants; variantId++) {
            if (variantId < metaData.prof.numVariants) {
                if (device == MVCNN::TargetDevice::TargetDevice_VPUX37XX) {
                    VPUX_THROW_WHEN(currentPos >= outputLen / sizeof(HwpDpu27Mode0Data_t),
                                    "HWP profiling index is out of range");

                    const HwpDpu27Mode0Data_t outputDpu =
                            reinterpret_cast<const HwpDpu27Mode0Data_t*>(output)[currentPos];
                    rawRecords.push_back(std::make_shared<RawProfilingDPUHW27Record>(
                            outputDpu, metaData.meta.taskName, metaData.meta.layerName, metaData.meta.layerType,
                            taskMetadata.waitBarriers, taskMetadata.updateBarriers, metaData.meta.clusterId, variantId,
                            currentPos));
                } else {
                    VPUX_THROW_WHEN(currentPos >= outputLen / sizeof(SwDpuData_t),
                                    "SW profiling index is out of range");

                    const SwDpuData_t outputDpu = reinterpret_cast<const SwDpuData_t*>(output)[currentPos];
                    rawRecords.push_back(std::make_shared<RawProfilingDPUSWRecord>(
                            outputDpu, metaData.meta.taskName, metaData.meta.layerName, metaData.meta.layerType,
                            taskMetadata.waitBarriers, taskMetadata.updateBarriers, metaData.meta.clusterId, variantId,
                            currentPos));
                }
                std::dynamic_pointer_cast<ThrowableAssertMixin>(rawRecords.back())->checkDataOrDie();
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
    VPUX_THROW_UNLESS(outputLen == NUM_WORKPOINTS * sizeof(WorkpointConfiguration_t),
                      "Profiling blob should have only {0} workpoint configurations", NUM_WORKPOINTS);
    const auto* workpointsPtr = reinterpret_cast<const WorkpointConfiguration_t*>(output);

    std::vector<std::pair<WorkpointConfiguration_t, size_t>> workpoints;
    for (size_t i = 0; i < NUM_WORKPOINTS; ++i) {
        workpoints.emplace_back(workpointsPtr[i], offset);
        offset += sizeof(WorkpointConfiguration_t);
    }

    return workpoints;
}

template <class DmaTaskType, class DpuTaskType, class SwTaskType>
RawProfilingData parseProfilingTaskLists(const std::map<ExecutorType, std::pair<uint32_t, uint32_t>>& offsets,
                                         MVCNN::TargetDevice device, const uint8_t* profData, TaskType type,
                                         const flatbuffers::Vector<flatbuffers::Offset<DmaTaskType>>* dmaTaskList,
                                         const flatbuffers::Vector<flatbuffers::Offset<DpuTaskType>>* dpuTaskList,
                                         const flatbuffers::Vector<flatbuffers::Offset<SwTaskType>>* swTaskList,
                                         const MVCNN::GraphFile* maybeGraphFile) {
    RawProfilingData rawProfData;

    for (const auto& p : offsets) {
        const auto offset = p.second.first;
        const auto length = p.second.second;

        switch (p.first) {
        case ExecutorType::DMA_SW: {
            rawProfData.dmaTasks = parseDmaSwTaskProfiling(dmaTaskList, profData + offset, length, device);
            rawProfData.parseOrder.emplace_back(ExecutorType::DMA_SW, offset);
            break;
        }
        case ExecutorType::DMA_HW: {
            rawProfData.dmaTasks = parseDmaHwTaskProfiling(maybeGraphFile, dmaTaskList, profData + offset, length);
            rawProfData.parseOrder.emplace_back(ExecutorType::DMA_HW, offset);
            break;
        }
        case ExecutorType::UPA: {
            if (type == TaskType::ALL || type == TaskType::DPU_SW) {
                rawProfData.swTasks = parseUPATaskProfiling(swTaskList, profData + offset, length);
                rawProfData.parseOrder.emplace_back(ExecutorType::UPA, offset);
            }
            break;
        }
        case ExecutorType::ACTSHAVE: {
            if (type == TaskType::ALL || type == TaskType::DPU_SW) {
                rawProfData.swTasks = parseActShaveTaskProfiling(swTaskList, profData + offset, length);
                rawProfData.parseOrder.emplace_back(ExecutorType::ACTSHAVE, offset);
            }
            break;
        }
        case ExecutorType::DPU: {
            if (type == TaskType::ALL || type == TaskType::DPU_SW) {
                rawProfData.dpuTasks = parseDPUTaskProfiling(dpuTaskList, profData + offset, length, device);
                rawProfData.parseOrder.emplace_back(ExecutorType::DPU, offset);
            }
            break;
        }
        case ExecutorType::WORKPOINT: {
            const auto isWorkpointAccessible = device == MVCNN::TargetDevice::TargetDevice_VPUX37XX;
            if (isWorkpointAccessible && type == TaskType::ALL && length != 0) {
                rawProfData.workpointsConfiguration = getWorkpointData(profData + offset, length, offset);
            }
            break;
        }
        case ExecutorType::NONE: {
            VPUX_THROW("None is not a valid profiling executor.");
        }
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
    if (pllMult < 10 || pllMult > 39) {
        vpux::Logger::global().warning("PLL multiplier '{0}' is out of [10; 39] range. MAX freq. setup will be used.",
                                       pllMult);
        return MAX_37XX_FREQ_SETUP;
    }
    const double VPU_TO_PLL_RATIO = 25.0;
    const double DPU_TO_VPU_RATIO = 4.0 / 3.0;

    const double vpuFreq = pllMult * VPU_TO_PLL_RATIO;
    const double dpuFreq = vpuFreq * DPU_TO_VPU_RATIO;
    return getFreqSetup37XXHelper(vpuFreq, dpuFreq, PROF_CLK_VALUE_MHZ);
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
        case MVCNN::TargetDevice::TargetDevice_VPUX311X:
            frcSpeedMhz = 700;
            break;
        default:
            VPUX_THROW("TargetDevice {0} is not supported ", EnumNameTargetDevice(graphFile->header()->device()));
        }
    }

    return frcSpeedMhz;
}

template <class FbType>
std::map<ExecutorType, std::pair<uint32_t, uint32_t>> getProfilingOffsets(const FbType* profBuffer,
                                                                          size_t actualBufferSize) {
    VPUX_THROW_UNLESS(profBuffer != nullptr, "Profiling buffer data must be not empty");
    VPUX_THROW_UNLESS(profBuffer->dimensions()->size() == 1, "Dimensions must be 1D tensor");

    const std::string outputName = profBuffer->name()->str();

    // Profiling buffer size is defined as tensor of shape <Nxui32>
    const uint32_t profSize = profBuffer->dimensions()->Get(0) * sizeof(uint32_t);
    VPUX_THROW_WHEN(uint32_t(actualBufferSize) < profSize,
                    "Actual buffer size is smaller than calculated. Expected {0}, but got {1}", profSize,
                    actualBufferSize);
    return parseProfilingOffsets(outputName, profSize);
}

std::map<ExecutorType, std::pair<uint32_t, uint32_t>> getProfilingOffsetsGf(const MVCNN::GraphFile* graphFile,
                                                                            size_t actualBufferSize) {
    const auto profilingOutputs = graphFile->header()->profiling_output();
    VPUX_THROW_UNLESS(profilingOutputs, "Blob contains no profiling_output");
    VPUX_THROW_UNLESS(profilingOutputs->size() == 1, "Blob must contain exactly one profiling output");

    const auto* profBuffer = profilingOutputs->Get(0);
    return getProfilingOffsets(profBuffer, actualBufferSize);
}

std::map<ExecutorType, std::pair<uint32_t, uint32_t>> vpux::profiling::parseProfilingOffsets(
        const std::string& profilingOutputName, uint32_t profilingOutputSize) {
    const std::map<std::string, ExecutorType> converter{
            {"dma", ExecutorType::DMA_SW}, {"dmahw", ExecutorType::DMA_HW},      {"dpu", ExecutorType::DPU},
            {"upa", ExecutorType::UPA},    {"actshave", ExecutorType::ACTSHAVE}, {"pll", ExecutorType::WORKPOINT},
            {"pad", ExecutorType::NONE}};

    const char delimiter = '_';
    std::stringstream sstream(profilingOutputName);
    std::string token;
    std::vector<std::string> tokens;
    while (std::getline(sstream, token, delimiter)) {
        tokens.push_back(token);
    }

    // Store starting offset and size of given profiling data type
    std::map<ExecutorType, std::pair<uint32_t, uint32_t>> offsets;
    uint32_t nextOffset = profilingOutputSize;

    for (auto it = tokens.crbegin(); it != tokens.crend(); ++it) {
        const ExecutorType executorEngine = converter.at(*it);
        ++it;
        uint32_t currentOffset = stoul(*it);
        offsets[executorEngine] = std::make_pair(currentOffset, static_cast<uint32_t>(nextOffset - currentOffset));
        nextOffset = currentOffset;
    }

    return offsets;
}

SummaryInfo vpux::profiling::getSummary(const RawData& profData, size_t profSize) {
    SummaryInfo summary{};
    summary.totalBufferSize = profSize;

    const MVCNN::TargetDevice device = profData.device;
    const auto offsets = profData.offsets;

    for (const auto& p : offsets) {
        SummaryInfo::SectionInfo* si = nullptr;
        switch (p.first) {
        case ExecutorType::DMA_SW: {
            si = &(summary.dmaInfo);
            if (device == MVCNN::TargetDevice::TargetDevice_VPUX37XX) {
                si->entrySize = sizeof(DMA27Data_t);
            } else {
                si->entrySize = sizeof(DMA20Data_t);
            }
            break;
        }
        case ExecutorType::DMA_HW: {
            si = &(summary.dmaInfo);
            si->entrySize = sizeof(HwpDma40Data_t);
            break;
        }
        case ExecutorType::DPU: {
            si = &(summary.dpuInfo);
            if (device == MVCNN::TargetDevice::TargetDevice_VPUX37XX) {
                si->entrySize = sizeof(HwpDpu27Mode0Data_t);
            } else {
                si->entrySize = sizeof(SwDpuData_t);
            }
            break;
        }
        case ExecutorType::UPA: {
            si = &(summary.swInfo);
            si->entrySize = sizeof(UpaData_t);
            break;
        }
        case ExecutorType::ACTSHAVE: {
            si = &(summary.swInfo);
            si->entrySize = sizeof(ActShaveData_t);
            break;
        case ExecutorType::WORKPOINT: {
            si = &(summary.workpointInfo);
            si->entrySize = sizeof(WorkpointConfiguration_t);
            break;
        }
        }
        default:
            VPUX_THROW("Unknown executor type.");
            break;
        }
        si->bufferOffset = p.second.first;
        si->bufferSize = p.second.second;
        si->numOfTasks = si->bufferSize / si->entrySize;
    }

    VPUX_THROW_WHEN(summary.dmaInfo.bufferSize != (summary.dmaInfo.entrySize * summary.dmaInfo.numOfTasks),
                    "Wrong number of elements in DMA profiling buffer. {0} != {1} * {2}", summary.dmaInfo.bufferSize,
                    summary.dmaInfo.entrySize, summary.dmaInfo.numOfTasks);
    VPUX_THROW_WHEN(summary.dpuInfo.bufferSize != (summary.dpuInfo.entrySize * summary.dpuInfo.numOfTasks),
                    "Wrong number of elements in DPU profiling buffer. {0} != {1} * {2}", summary.dpuInfo.bufferSize,
                    summary.dpuInfo.entrySize, summary.dpuInfo.numOfTasks);
    VPUX_THROW_WHEN(summary.swInfo.bufferSize != (summary.swInfo.entrySize * summary.swInfo.numOfTasks),
                    "Wrong number of elements in SW profiling buffer. {0} != {1} * {2}", summary.swInfo.bufferSize,
                    summary.swInfo.entrySize, summary.swInfo.numOfTasks);
    VPUX_THROW_WHEN(summary.totalBufferSize != (summary.dmaInfo.bufferSize + summary.dpuInfo.bufferSize +
                                                summary.swInfo.bufferSize + summary.workpointInfo.bufferSize),
                    "Profiling buffer sizes doesn't match. Expected total size {0}.", summary.totalBufferSize);

    return summary;
}

std::vector<std::string> splitBySeparator(const std::string& s, char separator) {
    std::istringstream iss(s);
    std::string part;
    std::vector<std::string> parts;
    while (std::getline(iss, part, separator)) {
        parts.push_back(part);
    }
    return parts;
}

RawProfilingRecord::TokenizedTaskName RawProfilingRecord::tokenizeTaskName(const std::string& gfTaskName) {
    const auto numLocationSeparators = std::count(gfTaskName.begin(), gfTaskName.end(), LOCATION_ORIGIN_SEPARATOR);
    bool hasMalformedMeta = false;
    if (numLocationSeparators > 1) {
        vpux::Logger::global().warning("Malformed task name \"{0}\". Only one layer and metadata are expected, but got "
                                       "{1}. Extra information will be thrown out",
                                       gfTaskName, numLocationSeparators);
        hasMalformedMeta = true;
    }
    // Find first separator to locate name
    auto firstNameSepPos = gfTaskName.find(LOCATION_ORIGIN_SEPARATOR);
    VPUX_THROW_WHEN(firstNameSepPos == std::string::npos, "Unparsable task name: '{0}' (has no NAMESEP)", gfTaskName);
    auto layerName = gfTaskName.substr(0, firstNameSepPos);
    // Find last separator to locate profiling metadata(proftag, clusterId)
    auto lastNameSepPos = gfTaskName.rfind(LOCATION_ORIGIN_SEPARATOR);
    VPUX_THROW_WHEN(lastNameSepPos == std::string::npos, "Unparsable task name: '{0}' (has no NAMESEP)", gfTaskName);
    auto afterNameSep = gfTaskName.substr(lastNameSepPos + 1);
    std::vector<std::string> parts = splitBySeparator(afterNameSep, LOCATION_SEPARATOR);
    return {layerName, parts, hasMalformedMeta};
}

RawProfilingRecord::ParsedTaskName RawProfilingRecord::deserializeTaskName(const std::string& fullTaskName,
                                                                           const std::string& profPrefix) {
    const auto LOC_METADATA_SEPARATOR = '_';  // conventional separator used for attaching metadata to MLIR Locations
    const auto CLUSTER_ID_MARKER = "cluster";

    auto tokenized = tokenizeTaskName(fullTaskName);

    int clusterId = 0;
    std::string profTag;
    std::string layerType;
    std::string layerName = tokenized.layerName;
    std::string reconstructedTaskName = layerName;
    bool isFirst = true;

    for (const auto& token : tokenized.tokens) {
        VPUX_THROW_WHEN(token.empty(), "Empty task name token");

        auto parts = splitBySeparator(token, LOC_METADATA_SEPARATOR);
        auto partsNum = parts.size();

        if (partsNum >= 1 && parts[0] == profPrefix) {
            profTag = token;
            continue;
        } else if (partsNum > 1 && parts[partsNum - 2] == CLUSTER_ID_MARKER) {
            clusterId = stoi(parts[partsNum - 1]);
            continue;
        } else if (partsNum == 2 && parts[0] == vpux::LOCATION_LAYER_TYPE_PREFIX) {
            layerType = parts[1];
            if (tokenized.hasMalformedMeta) {
                layerType += "_META_PARSING_ERROR";
            }
        }

        // Reconstruct the original task name skipping the profiling tag and cluster id
        const auto separator = isFirst ? LOCATION_ORIGIN_SEPARATOR : LOCATION_SEPARATOR;
        reconstructedTaskName = reconstructedTaskName + separator + token;
        isFirst = false;
    }

    VPUX_THROW_WHEN(profTag.empty(), "Couldn't find prof marker when deserializing task name: {0}", fullTaskName);

    return {reconstructedTaskName, layerName, layerType, profTag, clusterId};
}

std::string RawProfilingRecord::getLayerName(const std::string& taskName) {
    return taskName.substr(0, taskName.rfind(LOCATION_ORIGIN_SEPARATOR));
}

template <class T>
typename RawProfilingDMARecord<T>::ParsedProf RawProfilingDMARecord<T>::parseProfilingTag(const std::string& profTag) {
    auto subparts = splitBySeparator(profTag, '_');
    VPUX_THROW_UNLESS(subparts[0] == PROFILING_DMA_TASK_END_PREFIX, "DMA profiling marker has an unexpected prefix");
    VPUX_THROW_UNLESS(subparts.size() == 2, "Malformed DMA task PROFTASKEND location, expected to contain 2 parts");

    int16_t curDmaId = stoi(subparts[1]);
    return {curDmaId};
}

template <class T>
bool RawProfilingDMARecord<T>::isTaskBegin(const std::string& fullTaskName) {
    auto sepPos = fullTaskName.rfind(LOCATION_ORIGIN_SEPARATOR);
    return fullTaskName.find(PROFILING_DMA_TASK_BEGIN_PREFIX, sepPos) != std::string::npos;
}

template <class T>
bool RawProfilingDMARecord<T>::isTaskWorkpointRead(const std::string& fullTaskName) {
    return fullTaskName == PROFILING_WORKPOINT_READ_ATTR;
}

template <class T>
typename RawProfilingDMARecord<T>::ParsedTaskNameProf RawProfilingDMARecord<T>::parseTaskName(
        const std::string& fullTaskName) {
    auto parsedName = RawProfilingRecord::deserializeTaskName(fullTaskName, PROFILING_DMA_TASK_END_PREFIX);
    auto parsedProf = parseProfilingTag(parsedName.profTag);
    return {parsedName, parsedProf};
}

RawProfilingDPURecord::ParsedProf RawProfilingDPURecord::parseProfilingTag(const std::string& profTag,
                                                                           int16_t clusterId, unsigned taskListId) {
    auto parts = splitBySeparator(profTag, '_');
    VPUX_THROW_UNLESS(parts[0] == PROFILING_PREFIX, "ACT profiling marker has an unexpected prefix");
    VPUX_THROW_UNLESS(parts.size() == 5, "Malformed DPU task PROF location, expected to have 5 parts");

    ParsedProf parsed;
    parsed.memoryId = stoi(parts[1]);
    parsed.bufferId = stoi(parts[2]);
    parsed.numClusters = stoi(parts[3]);
    parsed.taskId = taskListId;
    std::string dpuTasksDistribution = parts[4];

    auto seppos = dpuTasksDistribution.find("-");
    VPUX_THROW_WHEN(seppos == std::string::npos, "Malformed DPU task profiling location: {0}", profTag);
    parsed.maxVariants = stoi(dpuTasksDistribution.substr(0));
    std::string cleanTaskDistribution = dpuTasksDistribution.substr(seppos + 1);

    parsed.clusterId = clusterId;
    parsed.numVariants = -1;
    auto taskDistribution = splitBySeparator(cleanTaskDistribution, ',');
    parsed.numVariants = stoi(taskDistribution.at(parsed.clusterId));

    return parsed;
}

RawProfilingDPURecord::ParsedTaskNameProf RawProfilingDPURecord::parseTaskName(const std::string& fullTaskName,
                                                                               unsigned taskListId) {
    auto parsedName = RawProfilingRecord::deserializeTaskName(fullTaskName, PROFILING_PREFIX);
    auto parsedProf = parseProfilingTag(parsedName.profTag, parsedName.clusterId, taskListId);

    return {parsedName, parsedProf};
}

RawProfilingUPARecord::ParsedProf RawProfilingUPARecord::parseProfilingTag(const std::string& profTag) {
    auto subparts = splitBySeparator(profTag, '_');
    VPUX_THROW_UNLESS(subparts[0] == PROFILING_PREFIX, "UPA profiling marker has an unexpected prefix: {0}", profTag);
    VPUX_THROW_UNLESS(subparts.size() == 2, "Malformed UPA task PROF location, expected to have 2 parts");

    size_t currentPos = stoull(subparts[1]);
    return {currentPos};
}

RawProfilingUPARecord::ParsedTaskNameProf RawProfilingUPARecord::parseTaskName(const std::string& fullTaskName) {
    auto parsedName = RawProfilingRecord::deserializeTaskName(fullTaskName, PROFILING_PREFIX);
    auto parsedProf = parseProfilingTag(parsedName.profTag);

    return {parsedName, parsedProf};
}

RawProfilingACTRecord::ParsedProf RawProfilingACTRecord::parseProfilingTag(const std::string& profTag,
                                                                           int16_t clusterId) {
    auto subparts = splitBySeparator(profTag, '_');
    VPUX_THROW_UNLESS(subparts[0] == PROFILING_PREFIX, "ACT profiling marker has an unexpected prefix");
    VPUX_THROW_UNLESS(subparts.size() == 5, "Malformed ACT task PROF location, expected to have 5 args: {0}", profTag);

    ParsedProf parsed;
    parsed.clusterId = clusterId;
    parsed.inDdrOffset = stoi(subparts[1]);
    parsed.clusterSize = stoi(subparts[2]);
    parsed.inClusterOffset = stoi(subparts[3]);
    parsed.tileId = stoi(subparts[4]);

    return parsed;
}

RawProfilingACTRecord::ParsedTaskNameProf RawProfilingACTRecord::parseTaskName(const std::string& fullTaskName) {
    auto parsedName = RawProfilingRecord::deserializeTaskName(fullTaskName, PROFILING_PREFIX);
    auto parsedProf = parseProfilingTag(parsedName.profTag, parsedName.clusterId);

    return {parsedName, parsedProf};
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

bool RawProfilingData::hasWorkpointConfig() const {
    return !workpointsConfiguration.empty();
}

uint16_t RawProfilingData::getPllValueChecked(vpux::Logger& log) const {
    // PLL value from begin of inference
    const auto pllMultFirst = workpointsConfiguration.front().first.pllMultiplier;
    // PLL value from end of inference
    const auto pllMultLast = workpointsConfiguration.back().first.pllMultiplier;
    if (pllMultFirst != pllMultLast) {
        log.warning("Frequency changed during the inference: {0} != {1}", pllMultFirst, pllMultLast);
    }
    return pllMultFirst;
}

RawData getRawProfilingTasksGraphFile(const MVCNN::GraphFile* graphFile, const uint8_t* profData, size_t profSize,
                                      TaskType type) {
    const auto device = graphFile->header()->device();

    // Finding of corresponding task list //
    const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* dmaTaskList = nullptr;
    const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* dpuTaskList = nullptr;
    const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* swTaskList = nullptr;
    auto taskLists = graphFile->task_lists();
    VPUX_THROW_UNLESS(taskLists, "Blob contains no taskLists");
    for (const auto& taskListItem : *taskLists) {
        const auto content = taskListItem->content();
        if (content->size() == 0) {
            continue;
        }
        const auto task0_type = content->Get(0)->task_type();
        if (task0_type == MVCNN::SpecificTask_NNDMATask) {
            dmaTaskList = taskListItem->content();
        }
        if (task0_type == MVCNN::SpecificTask_NCE2Task) {
            dpuTaskList = taskListItem->content();
        }
        if ((task0_type == MVCNN::SpecificTask_UPALayerTask) || (task0_type == MVCNN::SpecificTask_ActKernelTask)) {
            swTaskList = taskListItem->content();
        }
    }

    // Finding offsets of different profiling type in the profiling output
    const auto offsets = getProfilingOffsetsGf(graphFile, profSize);

    RawProfilingData rawProfData =
            parseProfilingTaskLists(offsets, device, profData, type, dmaTaskList, dpuTaskList, swTaskList, graphFile);
    llvm::Optional<double> maybe30XXFreq;
    if (device == MVCNN::TargetDevice::TargetDevice_VPUX30XX || device == MVCNN::TargetDevice::TargetDevice_VPUX311X) {
        maybe30XXFreq = getNceFreq(graphFile);
    }
    return {rawProfData, device, maybe30XXFreq, offsets};
}

const elf::Reader<elf::Elf64>::Section getProfilingSection(const elf::Reader<elf::ELF_Bitness::Elf64>& reader) {
    for (size_t i = 0; i < reader.getSectionsNum(); ++i) {
        const auto section = reader.getSection(i);
        const std::string secName = section.getName();
        if (secName == ".profiling") {
            return section;
        }
    }
    VPUX_THROW("Cannot find profiling section in ELF");
}

RawData getRawProfilingTasksElf(const uint8_t* blobData, size_t blobSize, const uint8_t* profData, size_t profSize,
                                TaskType type) {
    const auto device = MVCNN::TargetDevice_VPUX37XX;

    elf::ElfDDRAccessManager elfAccess(blobData, blobSize);
    elf::Reader<elf::ELF_Bitness::Elf64> reader(&elfAccess);

    const auto profilingSection = getProfilingSection(reader);
    const auto sectionData = profilingSection.getData<uint8_t>();

    const auto profilingDataSchema = ProfilingFB::GetProfilingMeta(sectionData);
    VPUX_THROW_UNLESS(profilingDataSchema != nullptr, "Empty ProfilingMeta");

    const auto profilingBufferMeta = profilingDataSchema->profilingBuffer();
    const auto offsets = getProfilingOffsets(profilingBufferMeta, profSize);

    RawProfilingData rawProfData =
            parseProfilingTaskLists(offsets, device, profData, type, profilingDataSchema->dmaTasks(),
                                    profilingDataSchema->dpuTasks(), profilingDataSchema->actShaveTasks(), nullptr);

    return {rawProfData, device, /*maybe30XXFreq*/ {}, offsets};
}

RawData vpux::profiling::getRawProfilingTasks(const uint8_t* blobData, size_t blobSize, const uint8_t* profData,
                                              size_t profSize, TaskType type) {
    (void)blobSize;

    if ((nullptr == blobData) || (nullptr == profData)) {
        VPUX_THROW("Empty input data");
    }

    auto& log = vpux::Logger::global();
    if (elf::utils::checkELFMagic(blobData)) {
        log.trace("Using ELF parser");
        return getRawProfilingTasksElf(blobData, blobSize, profData, profSize, type);
    }

    const auto* graphFile = MVCNN::GetGraphFile(blobData);
    log.trace("Using GF parser");
    return getRawProfilingTasksGraphFile(graphFile, profData, profSize, type);
}

ClusterTaskArray groupClusterTasks(const RawProfilingRecords& rawTasks) {
    ClusterTaskArray clusterInfoTasks;

    // Grouping of variants into one invariant
    std::multimap<std::pair<std::string, size_t>, RawProfilingRecordPtr> groupedClustersInfo;
    for (const auto& task : rawTasks) {
        const auto clusteredTask = std::dynamic_pointer_cast<ClusteredAndTiledMixin>(task);
        const auto clusterId = clusteredTask->getClusterId();
        const auto key = std::make_pair(task->getOriginalName(), clusterId);
        groupedClustersInfo.insert(std::make_pair(key, task));
    }

    auto it = groupedClustersInfo.cbegin();
    while (it != groupedClustersInfo.cend()) {
        RawProfilingRecords variants;
        const auto groupingKey = it->first;
        std::string name = groupingKey.first;
        const auto clusterId = groupingKey.second;
        const auto execType = it->second->getExecutorType();

        while (it != groupedClustersInfo.cend() && it->first == groupingKey) {
            variants.push_back(it->second);
            ++it;
        }
        clusterInfoTasks.push_back(std::make_shared<InvariantRawRecord>(name, clusterId, variants, execType));
    }

    return clusterInfoTasks;
}

RawProfilingRecords groupTasks(const ClusterTaskArray& clusterInfoTasks) {
    RawProfilingRecords dpuInfoTasks;

    // Grouping of invariants to one NCE Task
    std::multimap<std::string, std::shared_ptr<ArrayRecord>> groupedDPUTasksInfo;
    for (const auto& task : clusterInfoTasks) {
        groupedDPUTasksInfo.insert(std::make_pair(task->getOriginalName(), task));
    }
    auto it = groupedDPUTasksInfo.cbegin();
    while (it != groupedDPUTasksInfo.cend()) {
        ClusterTaskArray clusters;
        RawProfilingRecords variants;

        const auto groupingKey = it->first;
        const auto execType = it->second->getExecutorType();
        while (it != groupedDPUTasksInfo.cend() && it->first == groupingKey) {
            clusters.push_back(it->second);
            ++it;

            const auto first = clusters.front();
            const auto last = clusters.back();
            checkSameBarriers(first, last, /*checkWaitBarriers=*/true);
            checkSameBarriers(first, last, /*checkWaitBarriers=*/false);
            const auto newVariants = last->getArray();
            variants.insert(variants.end(), newVariants.begin(), newVariants.end());
        }
        dpuInfoTasks.push_back(std::make_shared<ArrayRecord>(groupingKey, variants, execType));
    }
    return dpuInfoTasks;
}

// At parse time we don't know frequency for some platforms, so data is collected in cycles format. We need
// to determine frequency to convert from cycles to nanoseconds
std::vector<TaskInfo> convertRawTasksToTaskInfo(const RawData& profData, bool fpga, VerbosityLevel verbosity) {
    auto log = vpux::Logger::global();
    const auto rawTasks = profData.rawRecords;
    const auto device = profData.device;

    ClusterTaskArray dpuClusterInfoTasks = groupClusterTasks(rawTasks.dpuTasks);
    RawProfilingRecords dpuInfoTasks = groupTasks(dpuClusterInfoTasks);

    FrequenciesSetup frequenciesSetup;

    if (device == MVCNN::TargetDevice::TargetDevice_VPUX37XX) {
        uint16_t pllMult = 0;
        if (rawTasks.hasWorkpointConfig()) {
            pllMult = rawTasks.getPllValueChecked(log);
        } else {
            log.warning("Missed PLL value.");
        }
        log.trace("Got PLL value of '{0}' from binary", pllMult);

        frequenciesSetup = fpga ? getFpgaFreqSetup(device) : getFreqSetupFromPll(device, pllMult);
    } else if (device == MVCNN::TargetDevice::TargetDevice_VPUX311X ||
               device == MVCNN::TargetDevice::TargetDevice_VPUX30XX) {
        VPUX_THROW_UNLESS(profData.maybe30XXNceFreq.hasValue(), "Missed NCE frequency for 30XX device");
        frequenciesSetup.profClk = profData.maybe30XXNceFreq.getValue();
        frequenciesSetup.dmaBandwidth = Dma20Bandwidth;
    } else {
        VPUX_THROW("TargetDevice {0} is not supported ", MVCNN::EnumNameTargetDevice(device));
    }
    log.info("Frequency setup is profClk={0}MHz, vpuClk={1}MHz, dpuClk={2}MHz", frequenciesSetup.profClk,
             frequenciesSetup.vpuClk, frequenciesSetup.dpuClk);

    for (const auto& taskList : {rawTasks.dmaTasks, rawTasks.dpuTasks, dpuInfoTasks}) {
        for (const auto& task : taskList) {
            task->sanitize(log, frequenciesSetup);
        }
    }

    std::vector<TaskInfo> dmaTaskInfo;
    std::vector<TaskInfo> swTaskInfo;
    std::vector<TaskInfo> dpuTaskInfo;

    fillTaskInfoWithParsedRawRecords(dmaTaskInfo, rawTasks.dmaTasks, frequenciesSetup);
    fillTaskInfoWithParsedRawRecords(dpuTaskInfo, dpuInfoTasks, frequenciesSetup);

    RawProfilingRecords swInfoTasks = rawTasks.swTasks;
    llvm::Optional<ClusterTaskArray> actClusterInfoTasks;
    bool swTasksMayBeTiled = device == MVCNN::TargetDevice::TargetDevice_VPUX37XX;
    if (swTasksMayBeTiled) {
        actClusterInfoTasks = groupClusterTasks(rawTasks.swTasks);
        swInfoTasks = groupTasks(actClusterInfoTasks.getValue());
    }
    fillTaskInfoWithParsedRawRecords(swTaskInfo, swInfoTasks, frequenciesSetup);

    const auto earliestDpuNs = getEarliestTaskBegin(dpuTaskInfo);
    if (verbosity >= VerbosityLevel::MEDIUM) {
        fillTaskInfoWithParsedRawRecords(dpuTaskInfo, dpuClusterInfoTasks, frequenciesSetup);
        if (swTasksMayBeTiled) {
            fillTaskInfoWithParsedRawRecords(swTaskInfo, actClusterInfoTasks.getValue(), frequenciesSetup);
        }
    }
    if (verbosity >= VerbosityLevel::HIGH) {
        fillTaskInfoWithParsedRawRecords(dpuTaskInfo, rawTasks.dpuTasks, frequenciesSetup);
        fillTaskInfoWithParsedRawRecords(swTaskInfo, rawTasks.swTasks, frequenciesSetup);
    }

    const auto earliestDmaNs = getEarliestTaskBegin(dmaTaskInfo);
    const auto earliestSwNs = getEarliestTaskBegin(swTaskInfo);

    log.trace("Earliest DMA: {0}", earliestDmaNs);
    log.trace("Earliest DPU: {0}", earliestDpuNs);
    log.trace("Earliest SW : {0}", earliestSwNs);

    if (!dmaTaskInfo.empty()) {
        adjustZeroPoint(dmaTaskInfo, 0, earliestDmaNs.getValue());
    }

    if (!dpuTaskInfo.empty()) {
        int64_t dma2dpuOffset = 0;
        if (!frequenciesSetup.hasSharedDmaDpuCounter) {
            const auto timersShift = getDMA2OtherTimersShift(rawTasks.dmaTasks, dpuInfoTasks, frequenciesSetup,
                                                             SynchronizationPointKind::STRICT_DMA_TO_DPU, log);
            log.trace("Timers DMA2DPU difference: {0}", timersShift);
            dma2dpuOffset = getTimersOffset(timersShift, earliestDmaNs, earliestDpuNs.getValue());
        } else {
            // If DMA profiling enabled difference is 0, otherwise setting to earliest task without call to
            // getTimersOffset to avoid counting twice
            dma2dpuOffset = earliestDmaNs.hasValue() ? 0 : -static_cast<int64_t>(earliestDpuNs.getValue());
        }
        adjustZeroPoint(dpuTaskInfo, dma2dpuOffset, earliestDmaNs);
        // Order DPU tasks by time to make tests more stable
        std::sort(dpuTaskInfo.begin(), dpuTaskInfo.end(), [](const TaskInfo& a, const TaskInfo& b) {
            if (a.start_time_ns != b.start_time_ns) {
                return a.start_time_ns < b.start_time_ns;
            } else {
                return std::strcmp(a.name, b.name) < 0;
            }
        });
    }

    if (!swTaskInfo.empty()) {
        int64_t dma2SwOffset = 0;
        if (!frequenciesSetup.hasSharedDmaSwCounter) {
            const auto dma2UpaTimerShift = getDMA2OtherTimersShift(
                    rawTasks.dmaTasks, rawTasks.swTasks, frequenciesSetup, SynchronizationPointKind::DMA_TO_UPA, log);
            log.trace("Timers DMA2UPA difference: {0}", dma2UpaTimerShift);
            dma2SwOffset = getTimersOffset(dma2UpaTimerShift, earliestDmaNs, earliestSwNs.getValue());
        } else {
            // If DMA profiling enabled difference is 0, otherwise setting to earliest task without call to
            // getTimersOffset to avoid counting twice
            dma2SwOffset = earliestDmaNs.hasValue() ? 0 : -static_cast<int64_t>(earliestSwNs.getValue());
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
                                                   size_t profSize, TaskType type, VerbosityLevel verbosity,
                                                   bool fpga) try {
    const auto rawProfData = getRawProfilingTasks(blobData, blobSize, profData, profSize, type);
    return convertRawTasksToTaskInfo(rawProfData, fpga, verbosity);
} catch (const std::exception& ex) {
    vpux::Logger::global().info("Post-processing error: '{0}'", ex.what());
    VPUX_THROW("Profiling post-processing failed");
}

std::vector<LayerInfo> vpux::profiling::getLayerInfo(const uint8_t* blobData, size_t blobSize, const uint8_t* profData,
                                                     size_t profSize, bool fpga) {
    std::vector<TaskInfo> taskInfo =
            getTaskInfo(blobData, blobSize, profData, profSize, TaskType::ALL, VerbosityLevel::LOW, fpga);

    return getLayerInfo(taskInfo);
}

std::vector<LayerInfo> vpux::profiling::getLayerInfo(const std::vector<TaskInfo>& taskInfo) {
    std::vector<LayerInfo> layerInfo{};
    for (auto& task : taskInfo) {
        LayerInfo* layer;
        std::string taskName(task.name);
        if (taskName.find("cluster_") != std::string::npos) {
            // Skipping high verbose tasks with cluster/variant info
            continue;
        }

        std::string layerName = RawProfilingRecord::getLayerName(taskName);

        auto result = std::find_if(begin(layerInfo), end(layerInfo), [&](LayerInfo item) {
            return layerName == item.name;
        });
        if (result == end(layerInfo)) {
            LayerInfo info = LayerInfo();

            auto length = layerName.copy(info.name, sizeof(info.name) - 1);
            info.name[length] = '\0';

            const std::string layerTypeStr(task.layer_type);
            length = layerTypeStr.copy(info.layer_type, sizeof(info.layer_type) - 1);
            info.layer_type[length] = '\0';

            info.status = LayerInfo::layer_status_t::EXECUTED;
            info.start_time_ns = task.start_time_ns;
            info.duration_ns = 0;
            layerInfo.push_back(info);
            layer = &layerInfo.back();
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
        } else if (task.exec_type == TaskInfo::ExecType::SW) {
            layer->sw_ns += task.duration_ns;
        } else if (task.exec_type == TaskInfo::ExecType::DMA) {
            layer->dma_ns += task.duration_ns;
        }
    }

    return layerInfo;
}
