//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/utils/plugin/profiling_parser.hpp"
#include "vpux/utils/IE/prefix.hpp"
#include "vpux/utils/IE/profiling.hpp"

#include <map>
#include <sstream>
#include <string>

using namespace vpux::profiling;

// Container for conjucted storage of tasks of one format: RawProfilingRecordPtr/TaskInfo
template <class Storage>
struct TasksContainer {
    std::vector<Storage> dmaTasks;
    std::vector<Storage> dpuTasks;
    std::vector<Storage> swTasks;
};

using RawTasksContainer = TasksContainer<RawProfilingRecordPtr>;
// Main synchronization primitive. First contain tasks, that update barrier, Second - that wait for barrier
using SynchronizationPoint = std::pair<RawProfilingRecords, RawProfilingRecords>;
using SynchronizationPointsContainer = std::vector<SynchronizationPoint>;
using ClusterTaskArray = std::vector<std::shared_ptr<ArrayRecord>>;

// For counters synchronization may be used tasks which run on different engines, but share common barrier -
// synchronization point. We're assume, that synchronization is immidiate, so at the moment both counters show same time
// For 37XX we have to kind of sync. points - DMA->DPU and DPU-DMA. Some models have jitter which reduce accuracy of
// frequency/shift estimation, so aggregation of two algorithm may be used. BIDIR_MEAN - average of two basic
// algorithms, BIDIR_MAX - max frequency, because DVFS algorithm is more precise for low frequency and usually DMA2DPU
// == DPU2DMA, while for high frequency and compute-bound model algorithms may diverge(from expirements up to 3 pll
// values) BIDIR_MAX - DEFAULT algorithm
enum class SynchronizationAlgorithKind { DMA_TO_DPU, DPU_TO_DMA, BIDIR_MEAN, BIDIR_MAX };
enum class SynchronizationPointKind { DMA_TO_DPU, DPU_TO_DMA, DMA_TO_UPA };

struct DPUProfilingMeta {
    std::string taskName;
    int16_t taskId;
    int16_t memoryId;
    int32_t maxVariants;
    int16_t numVariants;
    int16_t clusterId;
    int16_t bufferId;
    int16_t numClusters;

    // Profiling data is stored in blob in following order
    // Tasks which correspond to NCEClusterTasks with fewer number of clusters located earlier
    // Within one segment, which correspond to same amount of clusters, buffers located sequentially
    // Within one buffer tasks interleaved, but clusters  sequential, i.g.
    // Task-N cluster 0, Task-(N+1), cluster 0, Task-N cluster 1, Task-(N+1) cluster 1...
    uint64_t getOrderDescriptor() const {
        uint64_t descriptor = 0;
        for (uint16_t subDesc : {numClusters, bufferId, clusterId, memoryId}) {
            descriptor = (descriptor << 16) | subDesc;
        }
        return descriptor;
    }
};

// Same idea as for DPUProfilingMeta
struct ACTProfilingMeta {
    std::string taskName;
    size_t inDdrOffset;
    size_t clusterSize;
    size_t clusterId;
    size_t inClusterOffset;
    size_t tileId;

    size_t getResultingDDROffset() const {
        return inDdrOffset + clusterSize * clusterId + inClusterOffset;
    }
};

namespace {

const double PROF_CLK_VALUE_37XX = 38.4;
const double PLL_VPU_FREQUENCY_SCALE_37XX = 25.0;
const FrequenciesSetup HIGH_VCC_37XX_FREQUENCIES_SETUP = {975.0, 1300.0, PROF_CLK_VALUE_37XX, DMA27Bandwidth};

FrequenciesSetup getFrequencySetupFromPll(size_t pllValue) {
    const double VPU_TO_DPU_CLK_RATIO = 4.0 / 3.0;
    const double vpuClk = pllValue * PLL_VPU_FREQUENCY_SCALE_37XX;
    const double dpuClk = vpuClk * VPU_TO_DPU_CLK_RATIO;
    return {vpuClk, dpuClk, PROF_CLK_VALUE_37XX, DMA27Bandwidth};
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
        case ExecutorType::DMA:
            return "DMA";
        case ExecutorType::DPU:
            return "DPU";
        case ExecutorType::UPA:
            return "UPA";
        case ExecutorType::ACTSHAVE:
            return "ACTSHAVE";
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

    for (const auto& tasksGroup : {taskGroup1, taskGroup2}) {
        for (const auto& task : tasksGroup) {
            for (const auto waitBarrier : task->getWaitBarriers()) {
                barrierSuccessors.insert(std::make_pair(waitBarrier, task));
                waitBarriers.insert(waitBarrier);
            }
            for (const auto updateBarrier : task->getUpdateBarriers()) {
                barrierPredecessors.insert(std::make_pair(updateBarrier, task));
                updateBarriers.insert(updateBarrier);
            }
        }
    }

    ExecutorType predecessorExecType = ExecutorType::DMA;
    ExecutorType successorExecType = ExecutorType::DPU;
    if (pointKind == SynchronizationPointKind::DPU_TO_DMA) {
        std::swap(predecessorExecType, successorExecType);
    } else if (pointKind == SynchronizationPointKind::DMA_TO_UPA) {
        successorExecType = ExecutorType::UPA;
    }

    // Possible synchronization points occurs on covered from both directions barriers
    const auto commonBarriers = RawProfilingRecord::getBarriersIntersection(waitBarriers, updateBarriers);
    SynchronizationPointsContainer synchronizationPoints;
    for (const auto& commonBarrier : commonBarriers) {
        RawProfilingRecords predecessors =
                getRelatedTasksOfKind(commonBarrier, barrierPredecessors, predecessorExecType);
        RawProfilingRecords successors = getRelatedTasksOfKind(commonBarrier, barrierSuccessors, successorExecType);
        if (!predecessors.empty() && !successors.empty()) {
            synchronizationPoints.push_back(std::make_pair(predecessors, successors));
        }
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

double computeVariance(const std::vector<double>& vec) {
    const auto N = vec.size();
    const double mean = std::accumulate(vec.begin(), vec.end(), 0.) / N;
    const auto variance = std::accumulate(vec.begin(), vec.end(), 0.,
                                          [=](double cumSum, double x) {
                                              return cumSum + (x - mean) * (x - mean);
                                          }) /
                          N;
    return variance;
}

// Algorithm to determine frequency from synchronization points. Main idea is that when we choose correct frequency -
// shifts for any synchronization point will be roughly the same. If we choose wrong frequency - duration of NCE tasks
// will be scaled and shift will increase or decrease with execution time. As measure of correctness algorithm use
// variance. For wrong frequency variance will be greater
std::pair<FrequenciesSetup, size_t> findFrequencySetup37XXFromSyncPoints(
        const SynchronizationPointsContainer& syncPoints, vpux::Logger& log) {
    // According to specification PLL have range from 15 to 39
    // But sometimes for some reason it's 10 (???)
    const size_t pllMinValue = 10;
    const size_t pllMaxValue = 39;

    if (syncPoints.empty()) {
        log.warning(
                "Cannot find frequency because of lack of synchronization points. Imply 37XX high VCC frequencies.");
        return std::make_pair(HIGH_VCC_37XX_FREQUENCIES_SETUP, pllMaxValue);
    }

    FrequenciesSetup bestFrequency;
    size_t bestPllValue = 0;
    double minVariance = std::numeric_limits<double>::max();
    for (size_t pllValue = pllMinValue; pllValue <= pllMaxValue; ++pllValue) {
        const FrequenciesSetup frequencyCandidate = getFrequencySetupFromPll(pllValue);

        const auto shiftEstimations = getBarrierShiftEstimations(syncPoints, frequencyCandidate, log);
        const auto variance = computeVariance(shiftEstimations);

        log.trace("PLL = {0}. Variance = {1}", pllValue, variance);
        if (variance < minVariance) {
            bestFrequency = frequencyCandidate;
            bestPllValue = pllValue;
            minVariance = variance;
        }
    }
    log.trace("PLL value: {0}", bestPllValue);
    return std::make_pair(bestFrequency, bestPllValue);
}

FrequenciesSetup findFrequencyForBidirAlg(const SynchronizationPointsContainer& dma2dpuPoints,
                                          const SynchronizationPointsContainer& dpu2dmaPoints,
                                          SynchronizationAlgorithKind algKind, vpux::Logger& log) {
    VPUX_THROW_WHEN(
            algKind == SynchronizationAlgorithKind::DPU_TO_DMA || algKind == SynchronizationAlgorithKind::DMA_TO_DPU,
            "Cannot pass one way algorithm in bidir function");
    const auto dma2dpuFreqResult = findFrequencySetup37XXFromSyncPoints(dma2dpuPoints, log);
    const auto dpu2dmaFreqResult = findFrequencySetup37XXFromSyncPoints(dpu2dmaPoints, log);
    log.trace("DPU2DMA PLL = {0}", dpu2dmaFreqResult.second);
    log.trace("DMA2DPU PLL = {0}", dma2dpuFreqResult.second);

    const auto mean = [](double a, double b) -> double {
        return (a + b) / 2.0;
    };
    const auto max = [](double a, double b) -> double {
        return std::max(a, b);
    };
    const auto dpu2dmaFreq = dpu2dmaFreqResult.first;
    const auto dma2dpuFreq = dma2dpuFreqResult.first;
    const auto reducer = (algKind == SynchronizationAlgorithKind::BIDIR_MAX) ? max : mean;
    return {reducer(dpu2dmaFreq.vpuClk, dma2dpuFreq.vpuClk), reducer(dpu2dmaFreq.dpuClk, dma2dpuFreq.dpuClk),
            PROF_CLK_VALUE_37XX, DMA27Bandwidth};
}

FrequenciesSetup findFrequencySetup37XX(const RawProfilingRecords& dmaTasks, const RawProfilingRecords& dpuTasks,
                                        SynchronizationAlgorithKind algKind) {
    std::map<SynchronizationAlgorithKind, std::string> alg2str = {
            {SynchronizationAlgorithKind::BIDIR_MEAN, "BIDIR_MEAN"},
            {SynchronizationAlgorithKind::BIDIR_MAX, "BIDIR_MAX"},
            {SynchronizationAlgorithKind::DPU_TO_DMA, "DPU_TO_DMA"},
            {SynchronizationAlgorithKind::DMA_TO_DPU, "DMA_TO_DPU"},
    };
    auto log = vpux::Logger::global();
    log.setName("PLL determinator");
    log.trace("Using {0} algorithm", alg2str[algKind]);
    auto nestedLog = log.nest();
    bool isOneWayAlg =
            algKind == SynchronizationAlgorithKind::DPU_TO_DMA || algKind == SynchronizationAlgorithKind::DMA_TO_DPU;

    if (!isOneWayAlg) {
        const auto dpu2dmaSyncPoints =
                findSynchronizationPoints(dmaTasks, dpuTasks, SynchronizationPointKind::DPU_TO_DMA);
        if (dpu2dmaSyncPoints.empty()) {
            log.trace("Cannot use {0} algorithm because of lack of synchronization points of DPU2DMA kind. "
                      "DMA2DPU algorith will be used instead.",
                      alg2str[algKind]);
            return findFrequencySetup37XX(dmaTasks, dpuTasks, SynchronizationAlgorithKind::DMA_TO_DPU);
        }

        const auto dma2dpuSyncPoints =
                findSynchronizationPoints(dmaTasks, dpuTasks, SynchronizationPointKind::DMA_TO_DPU);
        if (dma2dpuSyncPoints.empty()) {
            log.trace("Cannot use {0} algorithm because of lack of synchronization points of DMA2DPU kind. "
                      "DPU2DMA algorith will be used instead.",
                      alg2str[algKind]);
            return findFrequencySetup37XX(dmaTasks, dpuTasks, SynchronizationAlgorithKind::DPU_TO_DMA);
        }

        const auto finalFreq = findFrequencyForBidirAlg(dma2dpuSyncPoints, dpu2dmaSyncPoints, algKind, nestedLog);
        log.trace("Resulting F = {0} MHz, PLL ~ {1}", finalFreq.vpuClk,
                  finalFreq.vpuClk / PLL_VPU_FREQUENCY_SCALE_37XX);
        return finalFreq;
    }
    const auto syncPointKind = algKind == SynchronizationAlgorithKind::DPU_TO_DMA
                                       ? SynchronizationPointKind::DPU_TO_DMA
                                       : SynchronizationPointKind::DMA_TO_DPU;
    const auto syncPoints = findSynchronizationPoints(dmaTasks, dpuTasks, syncPointKind);
    return findFrequencySetup37XXFromSyncPoints(syncPoints, nestedLog).first;
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
static int64_t getTimersOffset(const llvm::Optional<int64_t> maybeTaskTimerDiff,
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
    const auto firstDma = static_cast<int64_t>(maybeFirstDma.getValueOr(0));
    for (auto& task : taskInfo) {
        int64_t startTimeNs = task.start_time_ns - firstDma + timerDiff;
        task.start_time_ns = std::max(startTimeNs, (int64_t)0);
    }
};

void cleanTaskName(char* name) {
    if (!name) {
        return;
    }
    const auto size = strlen(name);
    if (!size) {
        return;
    }
    if (name[size - 1] == vpux::ORIGINAL_NAME_SEPARATOR) {
        name[size - 1] = '\0';
    }
}

void cleanLayerName(std::string& name) {
    const auto suffixPos = name.rfind(vpux::ORIGINAL_NAME_SEPARATOR);
    if (suffixPos != std::string::npos) {
        name.erase(suffixPos);
    }
}

void getProfilingMeta(const std::string& taskName, unsigned size, std::string* profilingMeta) {
    size_t pos = taskName.length();
    for (size_t i = 0; i < size; i++) {
        size_t epos = pos;
        pos = taskName.rfind('_', epos);
        if (pos == std::string::npos) {
            break;
        }
        profilingMeta[size - 1 - i] = taskName.substr(pos + 1, epos - pos);
        pos--;
    }
}

std::string getPrefix(const std::string& str, const std::string& delimeter) {
    return str.substr(0, str.find(delimeter));
}

llvm::Optional<ACTProfilingMeta> parseActLocation(const std::string& fullTaskName) {
    const unsigned MIN_NUM_NAME_SEGMENTS = 5;
    const unsigned NUM_SEGMENTS_WITH_CLUSTER_AND_TILE_INFO = MIN_NUM_NAME_SEGMENTS + 2;
    const unsigned MAX_NUM_NAME_SEGMENTS = NUM_SEGMENTS_WITH_CLUSTER_AND_TILE_INFO;

    ACTProfilingMeta meta;
    meta.clusterId = 0;
    meta.tileId = 0;

    auto size = MIN_NUM_NAME_SEGMENTS;
    if (fullTaskName.find("_cluster_") != fullTaskName.npos) {
        size = NUM_SEGMENTS_WITH_CLUSTER_AND_TILE_INFO;
    }
    std::string segments[MAX_NUM_NAME_SEGMENTS];
    getProfilingMeta(fullTaskName, size, segments);
    if (segments[0] != "PROF") {
        return {};
    }

    if (size == NUM_SEGMENTS_WITH_CLUSTER_AND_TILE_INFO) {
        meta.clusterId = std::stoull(segments[6]);
    }
    meta.inDdrOffset = std::stoull(segments[1]);
    meta.clusterSize = std::stoull(segments[2]);
    meta.inClusterOffset = std::stoull(segments[3]);
    if (size == NUM_SEGMENTS_WITH_CLUSTER_AND_TILE_INFO) {
        meta.tileId = std::stoull(segments[4]);
    }

    std::string taskName = getPrefix(fullTaskName, "_PROF");
    if (!taskName.empty() && taskName[taskName.length() - 1] == '/') {
        taskName.pop_back();
    }
    meta.taskName = taskName;
    return meta;
}

DPUProfilingMeta parseDPULocation(const std::string& fullTaskName, unsigned taskListId) {
    const unsigned MIN_NUM_NAME_SEGMENTS = 5;
    const unsigned NUM_SEGMENTS_WITH_CLUSTER_INFO = 7;
    const unsigned MAX_NUM_NAME_SEGMENTS = NUM_SEGMENTS_WITH_CLUSTER_INFO;

    DPUProfilingMeta meta;
    auto size = MIN_NUM_NAME_SEGMENTS;
    int32_t currentClusterId = 0;
    if (fullTaskName.find("_cluster_") != fullTaskName.npos) {
        size = NUM_SEGMENTS_WITH_CLUSTER_INFO;
    }
    std::string segments[MAX_NUM_NAME_SEGMENTS];
    getProfilingMeta(fullTaskName, size, segments);
    if (size == NUM_SEGMENTS_WITH_CLUSTER_INFO) {
        currentClusterId = std::stoi(segments[6]);
    }

    std::string taskName;
    if (segments[0] == "PROF") {
        taskName = getPrefix(fullTaskName, "_PROF");
        if (!taskName.empty() && taskName[taskName.length() - 1] == '/') {
            taskName.pop_back();
        }
    } else {
        VPUX_THROW("Unparsable task name: {0}", fullTaskName);
    }

    int32_t clustersAmount = std::stoi(segments[3]);
    const auto dpuTasksDistribution = segments[4];
    std::string maxVariants = getPrefix(dpuTasksDistribution, "-");
    std::string cleanTaskDistribution = dpuTasksDistribution.substr(maxVariants.size() + 1);

    int32_t numVariants = -1;
    for (int32_t clusterId = 0; clusterId < clustersAmount; clusterId++) {
        auto prefix = getPrefix(cleanTaskDistribution, ",");
        if (clusterId == currentClusterId) {
            numVariants = std::stoi(prefix);
            break;
        }
        cleanTaskDistribution = cleanTaskDistribution.substr(prefix.size() + 1);
    }

    meta.taskName = taskName;
    meta.memoryId = std::stoi(segments[1]);
    meta.bufferId = std::stoi(segments[2]);
    meta.numClusters = clustersAmount;
    meta.clusterId = currentClusterId;
    meta.maxVariants = std::stoi(maxVariants);
    meta.numVariants = numVariants;
    meta.taskId = taskListId;
    return meta;
}

}  // namespace

static double getNceFreq(const MVCNN::GraphFile* graphFile) {
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

static std::map<ExecutorType, std::pair<uint32_t, uint32_t>> getProfilingOffsets(const MVCNN::GraphFile* graphFile,
                                                                                 size_t actualBufferSize) {
    auto profilingOutputs = graphFile->header()->profiling_output();
    VPUX_THROW_UNLESS(profilingOutputs, "Blob contains no profiling_output");
    VPUX_THROW_UNLESS(profilingOutputs->size() == 1, "Blob must contain exactly one profiling output");

    const std::map<std::string, ExecutorType> converter{{"dma", ExecutorType::DMA},
                                                        {"dpu", ExecutorType::DPU},
                                                        {"upa", ExecutorType::UPA},
                                                        {"actshave", ExecutorType::ACTSHAVE}};

    const char delimiter = '_';
    auto* profBuffer = profilingOutputs->Get(0);
    const std::string outputName = profBuffer->name()->str();
    // Profiling buffer size is defined as tensor of shape <Nxui32>
    const auto profSize = profBuffer->dimensions()->Get(0) * sizeof(uint32_t);
    VPUX_THROW_WHEN(actualBufferSize < profSize,
                    "Actual buffer size is smaller than calculated. Expected {0}, but got {1}", profSize,
                    actualBufferSize);

    std::stringstream sstream(outputName);
    std::string token;
    std::vector<std::string> tokens;
    while (std::getline(sstream, token, delimiter)) {
        tokens.push_back(token);
    }

    // Store starting offset and size of given profiling data type
    std::map<ExecutorType, std::pair<uint32_t, uint32_t>> offsets;
    uint32_t nextOffset = static_cast<uint32_t>(profSize);

    for (auto it = tokens.crbegin(); it != tokens.crend(); ++it) {
        const ExecutorType executorEngine = converter.at(*it);
        ++it;
        uint32_t currentOffset = std::stoul(*it);
        offsets[executorEngine] = std::make_pair(currentOffset, static_cast<uint32_t>(nextOffset - currentOffset));
        nextOffset = currentOffset;
    }

    return offsets;
}

SummaryInfo vpux::profiling::getSummary(const uint8_t* blobData, size_t profSize) {
    SummaryInfo summary{};
    summary.totalBufferSize = profSize;

    const auto* graphFile = MVCNN::GetGraphFile(blobData);
    const MVCNN::TargetDevice device = graphFile->header()->device();
    const std::map<ExecutorType, std::pair<uint32_t, uint32_t>> offsets = getProfilingOffsets(graphFile, profSize);

    for (const auto& p : offsets) {
        SummaryInfo::SectionInfo* si = nullptr;
        switch (p.first) {
        case ExecutorType::DMA: {
            si = &(summary.dmaInfo);
            if (device == MVCNN::TargetDevice::TargetDevice_VPUX37XX) {
                si->entrySize = sizeof(DebugInfo::Raw::dma27);
            } else {
                si->entrySize = sizeof(DebugInfo::Raw::dma20);
            }
            break;
        }
        case ExecutorType::DPU: {
            si = &(summary.dpuInfo);
            if (device == MVCNN::TargetDevice::TargetDevice_VPUX37XX) {
                si->entrySize = sizeof(HwpDpuMode0Data_t);
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
    VPUX_THROW_WHEN(summary.totalBufferSize !=
                            (summary.dmaInfo.bufferSize + summary.dpuInfo.bufferSize + summary.swInfo.bufferSize),
                    "Profiling buffer sizes doesn't match. Expected total size {0}.", summary.totalBufferSize);

    return summary;
}

static std::vector<DebugInfo> parseDebugDMATaskProfiling(
        const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* dmaTaskList, const void* output, size_t outputLen,
        MVCNN::TargetDevice device) {
    if (dmaTaskList == nullptr) {
        return {};
    }

    auto log = vpux::Logger::global();

    RecordType rt;
    if (device != MVCNN::TargetDevice::TargetDevice_VPUX37XX) {
        rt = RecordType::DMA20;
    } else {
        rt = RecordType::DMA27;
    }

    std::vector<DebugInfo> profInfo;
    for (unsigned dmaTaskListId = 0; dmaTaskListId < (*dmaTaskList).size(); dmaTaskListId++) {
        auto task = (*dmaTaskList)[dmaTaskListId];
        if ((task->task_as_NNDMATask()->src()->name()->str() == "profilingInput:0") ||
            (task->task_as_NNDMATask()->src()->locale() == MVCNN::MemoryLocation_AbsoluteAddr)) {
            auto taskName = task->name()->str();

            std::string profilingMeta[3];  // PROFTASKEND_curDmaId_layerNum
            getProfilingMeta(taskName, 3, profilingMeta);

            if ((profilingMeta[2] == "PROFTASKBEGIN") || (profilingMeta[2] == "PROFBEGIN")) {
                continue;
            }

            const int layerNumber = stoi(profilingMeta[2]);
            const unsigned int beginDmaId = stoi(profilingMeta[1]);
            const unsigned int endDmaId = layerNumber * 2 - 1;

            DebugInfo profInfoItem = DebugInfo();
            profInfoItem.name = taskName.substr(0, taskName.find("_PROF"));
            profInfoItem.recordType = rt;

            if (device != MVCNN::TargetDevice::TargetDevice_VPUX37XX) {
                if ((beginDmaId >= outputLen / sizeof(profInfoItem.raw.dma20)) ||
                    (endDmaId >= outputLen / sizeof(profInfoItem.raw.dma20))) {
                    log.error("DMA index out of bounds!");
                    continue;
                }
                auto outputBin = reinterpret_cast<const uint32_t*>(output);
                profInfoItem.offset = beginDmaId * sizeof(profInfoItem.raw.dma20);
                profInfoItem.raw.dma20 = outputBin[beginDmaId];
                profInfo.push_back(profInfoItem);
                profInfoItem.offset = endDmaId * sizeof(profInfoItem.raw.dma20);
                profInfoItem.raw.dma20 = outputBin[endDmaId];
                profInfo.push_back(profInfoItem);
            } else {
                if ((beginDmaId >= outputLen / sizeof(profInfoItem.raw.dma27)) ||
                    (endDmaId >= outputLen / sizeof(profInfoItem.raw.dma27))) {
                    log.error("DMA index out of bounds!");
                    continue;
                }
                auto outputBin = reinterpret_cast<const uint64_t*>(output);
                profInfoItem.offset = beginDmaId * sizeof(profInfoItem.raw.dma27);
                profInfoItem.raw.dma27 = outputBin[beginDmaId];
                profInfo.push_back(profInfoItem);
                profInfoItem.offset = endDmaId * sizeof(profInfoItem.raw.dma27);
                profInfoItem.raw.dma27 = outputBin[endDmaId];
                profInfo.push_back(profInfoItem);
            }
        }
    }
    return profInfo;
}

static void parseDMATaskProfiling(RawProfilingRecords& rawRecords,
                                  const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* dmaTaskList,
                                  const void* output, size_t outputLen, MVCNN::TargetDevice device) {
    if (dmaTaskList == nullptr) {
        return;
    }

    uint64_t overflowShift = 0;
    uint32_t lastTime = 0;

    std::vector<TaskInfo> profInfo;
    size_t totalDmaTasks = 0;
    if (device != MVCNN::TargetDevice::TargetDevice_VPUX37XX) {
        totalDmaTasks = outputLen / sizeof(uint32_t);
    } else {
        totalDmaTasks = outputLen / sizeof(uint64_t);
    }

    BarriersSet lastProfilingRecordWaitBarriers;
    uint32_t foundDmaTasks = 0;
    for (unsigned dmaTaskListId = 0; dmaTaskListId < (*dmaTaskList).size(); dmaTaskListId++) {
        auto task = (*dmaTaskList)[dmaTaskListId];
        if ((task->task_as_NNDMATask()->src()->name()->str() == "profilingInput:0") ||
            (task->task_as_NNDMATask()->src()->locale() == MVCNN::MemoryLocation_AbsoluteAddr)) {
            auto taskName = task->name()->str();
            std::string profilingMeta[3];
            getProfilingMeta(taskName, 3, profilingMeta);

            if ((profilingMeta[2] != "PROFTASKBEGIN") && (profilingMeta[2] != "PROFBEGIN")) {
                foundDmaTasks += 2;
                unsigned layerNumber = 0;

                layerNumber = stoi(profilingMeta[2]);
                unsigned lastDMAid = stoi(profilingMeta[1]);
                auto currentDMAid = layerNumber * 2 - 1;

                taskName = taskName.substr(0, taskName.find("_PROF"));
                const auto updateBarriers = getBarriersFromTask(task, /*waitBarriers=*/false);

                VPUX_THROW_UNLESS((currentDMAid < totalDmaTasks) && (lastDMAid < totalDmaTasks),
                                  "Can't process DMA profiling data.");
                if (device != MVCNN::TargetDevice::TargetDevice_VPUX37XX) {
                    auto outputBin = reinterpret_cast<const uint32_t*>(output);

                    // Catch overflow and increase overflow shift for absolute start time
                    if (lastTime > 0x7F000000 && outputBin[lastDMAid] < 0x7F000000) {
                        overflowShift += 0x100000000;
                    }
                    lastTime = outputBin[lastDMAid];

                    rawRecords.push_back(std::make_shared<RawProfilingDMA20Record>(
                            outputBin[lastDMAid], outputBin[currentDMAid], taskName, lastProfilingRecordWaitBarriers,
                            updateBarriers, overflowShift));
                } else {
                    if ((currentDMAid >= outputLen / sizeof(uint64_t)) || (lastDMAid >= outputLen / sizeof(uint64_t))) {
                        continue;
                    }
                    auto outputBin = reinterpret_cast<const uint64_t*>(output);
                    rawRecords.push_back(std::make_shared<RawProfilingDMA27Record>(
                            outputBin[lastDMAid], outputBin[currentDMAid], taskName, lastProfilingRecordWaitBarriers,
                            updateBarriers));
                }
            } else {
                lastProfilingRecordWaitBarriers = getBarriersFromTask(task, /*waitBarriers=*/true);
            }
        }
    }
    VPUX_THROW_UNLESS(totalDmaTasks == foundDmaTasks, "Unexpected number of DMA tasks in profiling data");
}

static std::vector<DebugInfo> parseDebugUPATaskProfiling(
        const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* upaTaskList, const void* output,
        size_t outputLen) {
    if (upaTaskList == nullptr) {
        return {};
    }

    auto log = vpux::Logger::global();
    auto outputUpa = reinterpret_cast<const UpaData_t*>(output);

    std::vector<DebugInfo> profInfo;
    for (unsigned upaTaskListId = 0; upaTaskListId < upaTaskList->size(); upaTaskListId++) {
        auto task = (*upaTaskList)[upaTaskListId];
        auto taskName = task->name()->str();
        std::string profilingMeta[2];
        getProfilingMeta(taskName, 2, profilingMeta);

        if (profilingMeta[0] == "PROF") {
            taskName = taskName.substr(0, taskName.find("_PROF"));
            if (!taskName.empty() && taskName[taskName.length() - 1] == '/') {
                taskName.pop_back();
            }

            const unsigned currentPos = stoi(profilingMeta[1]);
            if (currentPos >= outputLen / sizeof(UpaData_t) ||
                (outputUpa[currentPos].begin == 0 && outputUpa[currentPos].end == 0)) {
                log.error("Can't process UPA profiling data.");
                continue;
            }

            DebugInfo profInfoItem = DebugInfo();
            profInfoItem.name = taskName;
            profInfoItem.offset = upaTaskListId * sizeof(UpaData_t);
            profInfoItem.recordType = RecordType::SW_UPA;
            profInfoItem.raw.upa = outputUpa[currentPos];
            profInfo.push_back(profInfoItem);
        }
    }
    return profInfo;
}

static void parseUPATaskProfiling(RawProfilingRecords& rawRecords,
                                  const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* upaTaskList,
                                  const void* output, size_t outputLen) {
    if (upaTaskList == nullptr) {
        return;
    }

    auto outputUpa = reinterpret_cast<const UpaData_t*>(output);
    const size_t totalUpaTasks = outputLen / sizeof(UpaData_t);
    size_t foundUpaTasks = 0;

    for (unsigned upaTaskListId = 0; upaTaskListId < upaTaskList->size(); upaTaskListId++) {
        auto task = (*upaTaskList)[upaTaskListId];
        auto taskName = task->name()->str();
        std::string profilingMeta[2];
        getProfilingMeta(taskName, 2, profilingMeta);

        if (profilingMeta[0] == "PROF") {
            taskName = taskName.substr(0, taskName.find("_PROF"));
            if (!taskName.empty() && taskName[taskName.length() - 1] == '/') {
                taskName.pop_back();
            }
            unsigned currentPos = stoi(profilingMeta[1]);
            VPUX_THROW_UNLESS(currentPos < outputLen / sizeof(UpaData_t),
                              "Unexpected end of blob in UPA profiling data.");
            foundUpaTasks++;

            auto softLayer = task->task_as_UPALayerTask();
            std::string layerName;
            if (softLayer != nullptr) {
                const char* typeName = EnumNameSoftwareLayerParams(softLayer->softLayerParams_type());
                if (typeName != nullptr) {
                    layerName = typeName;
                }
            }
            rawRecords.push_back(
                    std::make_shared<RawProfilingUPARecord>(outputUpa[currentPos], taskName, task, layerName));
            std::dynamic_pointer_cast<ThrowableAssertMixin>(rawRecords.back())->checkDataOrDie();
        }
    }
    VPUX_THROW_UNLESS(totalUpaTasks == foundUpaTasks, "Unexpected number of UPA tasks in profiling data");
}

static std::vector<DebugInfo> parseDebugActShaveTaskProfiling(
        const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* shaveTaskList, const void* output,
        size_t outputLen) {
    if (shaveTaskList == nullptr) {
        return {};
    }

    auto log = vpux::Logger::global();
    const ActShaveData_t* outputShave = reinterpret_cast<const ActShaveData_t*>(output);
    const size_t numOfActShaveTasks = outputLen / sizeof(ActShaveData_t);

    std::vector<DebugInfo> profInfo;
    for (unsigned asTaskListId = 0; asTaskListId < shaveTaskList->size(); ++asTaskListId) {
        const auto& task = (*shaveTaskList)[asTaskListId];
        auto taskName = task->name()->str();
        const auto maybeActMeta = parseActLocation(taskName);

        if (maybeActMeta.hasValue()) {
            const auto actMeta = maybeActMeta.getValue();
            size_t currentPos = actMeta.getResultingDDROffset();

            DebugInfo profInfoItem = DebugInfo();
            profInfoItem.name = taskName;
            profInfoItem.offset = asTaskListId * sizeof(ActShaveData_t);
            profInfoItem.recordType = RecordType::SW_ACT;

            if (currentPos >= numOfActShaveTasks) {
                log.error("Can't process ActShave profiling data, index out of range.");
                continue;
            }
            profInfoItem.raw.actShave = outputShave[currentPos];

            profInfo.push_back(profInfoItem);
        }
    }
    return profInfo;
}

static void parseActShaveTaskProfiling(RawProfilingRecords& rawRecords,
                                       const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* shaveTaskList,
                                       const void* output, size_t outputLen) {
    if (shaveTaskList == nullptr) {
        return;
    }

    const ActShaveData_t* outputShave = reinterpret_cast<const ActShaveData_t*>(output);
    const size_t numOfActShaveTasks = outputLen / sizeof(ActShaveData_t);
    size_t foundActShaveTasks = 0;

    for (const auto& task : *shaveTaskList) {
        auto taskName = task->name()->str();

        const auto maybeActMeta = parseActLocation(taskName);

        if (maybeActMeta.hasValue()) {
            const auto actMeta = maybeActMeta.getValue();
            size_t currentPos = actMeta.getResultingDDROffset();

            VPUX_THROW_UNLESS(currentPos < numOfActShaveTasks, "Unexpected end of blob in ACT section.");
            foundActShaveTasks++;
            rawRecords.push_back(std::make_shared<RawProfilingACTRecord>(outputShave[currentPos], actMeta.taskName,
                                                                         task, actMeta.clusterId, actMeta.tileId));

            std::dynamic_pointer_cast<ThrowableAssertMixin>(rawRecords.back())->checkDataOrDie();
        }
    }
    VPUX_THROW_UNLESS(foundActShaveTasks == shaveTaskList->size(), "All ActShave tasks should be profiled");
}

static std::vector<DebugInfo> parseDebugDPUTaskProfiling(
        const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* dpuTaskList, const void* output, size_t outputLen,
        MVCNN::TargetDevice device) {
    if (dpuTaskList == nullptr) {
        return {};
    }

    unsigned currentPos = 0;
    std::vector<DebugInfo> profInfo;
    for (unsigned dpu_taskListId = 0; dpu_taskListId < (*dpuTaskList).size(); dpu_taskListId++) {
        auto task = (*dpuTaskList)[dpu_taskListId];
        auto taskName = task->name()->str();
        auto metaData = parseDPULocation(taskName, dpu_taskListId);

        const auto clusterName =
                metaData.taskName + CLUSTER_LEVEL_PROFILING_SUFFIX + std::to_string(metaData.clusterId);

        DebugInfo profInfoItem = DebugInfo();
        if (device == MVCNN::TargetDevice::TargetDevice_VPUX37XX) {
            VPUX_THROW_WHEN(currentPos >= outputLen / sizeof(HwpDpuMode0Data_t), "HWP profiling index is out of range");
            profInfoItem.recordType = RecordType::DPU_HWP27;
        } else {
            VPUX_THROW_WHEN(currentPos >= outputLen / sizeof(SwDpuData_t), "SW profiling index is out of range");
            profInfoItem.recordType = RecordType::DPU_SW;
        }

        for (auto variantId = 0; variantId < metaData.maxVariants; variantId++) {
            if (variantId < metaData.numVariants) {
                if (device == MVCNN::TargetDevice::TargetDevice_VPUX37XX) {
                    profInfoItem.raw.hwpDpu = reinterpret_cast<const HwpDpuMode0Data_t*>(output)[currentPos];
                    profInfoItem.offset = currentPos * sizeof(HwpDpuMode0Data_t);
                } else {
                    profInfoItem.raw.swDpu = reinterpret_cast<const SwDpuData_t*>(output)[currentPos];
                    profInfoItem.offset = currentPos * sizeof(SwDpuData_t);
                }

                profInfoItem.name = clusterName + VARIANT_LEVEL_PROFILING_SUFFIX + std::to_string(variantId);
                profInfo.push_back(profInfoItem);
            }
            // increment of currentPos to walk over non-used data
            ++currentPos;
        }
    }

    return profInfo;
}

static void parseDPUTaskProfiling(RawProfilingRecords& rawRecords,
                                  const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* dpuTaskList,
                                  const void* output, size_t outputLen, MVCNN::TargetDevice device) {
    if (dpuTaskList == nullptr) {
        return;
    }

    // Ordering profiling tasks in ascending order
    std::map<uint64_t, DPUProfilingMeta> profilingMeta;
    for (unsigned dpu_taskListId = 0; dpu_taskListId < (*dpuTaskList).size(); dpu_taskListId++) {
        auto task = (*dpuTaskList)[dpu_taskListId];
        auto taskName = task->name()->str();
        auto meta = parseDPULocation(taskName, dpu_taskListId);
        profilingMeta.emplace(meta.getOrderDescriptor(), meta);
    }

    unsigned currentPos = 0;
    std::map<std::string, TaskInfo> profInfoAggregator;

    for (const auto& iter : profilingMeta) {
        const auto metaData = iter.second;
        const auto task = (*dpuTaskList)[metaData.taskId];

        for (auto variantId = 0; variantId < metaData.maxVariants; variantId++) {
            if (variantId < metaData.numVariants) {
                if (device == MVCNN::TargetDevice::TargetDevice_VPUX37XX) {
                    VPUX_THROW_WHEN(currentPos >= outputLen / sizeof(HwpDpuMode0Data_t),
                                    "HWP profiling index is out of range");

                    const HwpDpuMode0Data_t outputDpu = reinterpret_cast<const HwpDpuMode0Data_t*>(output)[currentPos];
                    rawRecords.push_back(std::make_shared<RawProfilingDPUHW27Record>(outputDpu, metaData.taskName, task,
                                                                                     metaData.clusterId, variantId));
                } else {
                    VPUX_THROW_WHEN(currentPos >= outputLen / sizeof(SwDpuData_t),
                                    "SW profiling index is out of range");

                    const SwDpuData_t outputDpu = reinterpret_cast<const SwDpuData_t*>(output)[currentPos];
                    rawRecords.push_back(std::make_shared<RawProfilingDPUSWRecord>(outputDpu, metaData.taskName, task,
                                                                                   metaData.clusterId, variantId));
                }
                std::dynamic_pointer_cast<ThrowableAssertMixin>(rawRecords.back())->checkDataOrDie();
            }
            // continue increment of currentPos to walk over non-used data
            ++currentPos;
        }
    }
}

std::vector<DebugInfo> vpux::profiling::getTaskInfoInDebugMode(const uint8_t* blobData, size_t blobSize,
                                                               const uint8_t* profData, size_t profSize,
                                                               TaskType type) {
    (void)blobSize;

    if ((nullptr == blobData) || (nullptr == profData)) {
        VPUX_THROW("Empty input data");
    }

    const auto* graphFile = MVCNN::GetGraphFile(blobData);

    // Finding of corresponding task list //
    const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* dmaTaskList = nullptr;
    const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* dpuTaskList = nullptr;
    const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* swTaskList = nullptr;
    auto taskLists = graphFile->task_lists();
    VPUX_THROW_UNLESS(taskLists, "Blob contains no taskLists");
    for (auto taskListItem : *taskLists) {
        auto task0_type = taskListItem->content()->Get(0)->task_type();
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
    const auto offsets = getProfilingOffsets(graphFile, profSize);
    const MVCNN::TargetDevice device = graphFile->header()->device();

    std::vector<DebugInfo> rawProfTask;

    for (const auto& p : offsets) {
        const auto offset = reinterpret_cast<const uint8_t*>(profData) + p.second.first;
        const auto length = p.second.second;

        switch (p.first) {
        case ExecutorType::DMA: {
            if (type == TaskType::ALL || type == TaskType::DMA) {
                std::vector<DebugInfo> dmaTasks = parseDebugDMATaskProfiling(dmaTaskList, offset, length, device);
                rawProfTask.insert(rawProfTask.end(), dmaTasks.begin(), dmaTasks.end());
            }
            break;
        }
        case ExecutorType::UPA: {
            if (type == TaskType::ALL || type == TaskType::DPU_SW) {
                std::vector<DebugInfo> upaTasks = parseDebugUPATaskProfiling(swTaskList, offset, length);
                rawProfTask.insert(rawProfTask.end(), upaTasks.begin(), upaTasks.end());
            }
            break;
        }
        case ExecutorType::ACTSHAVE: {
            if (type == TaskType::ALL || type == TaskType::DPU_SW) {
                std::vector<DebugInfo> actShaveTasks = parseDebugActShaveTaskProfiling(swTaskList, offset, length);
                rawProfTask.insert(rawProfTask.end(), actShaveTasks.begin(), actShaveTasks.end());
            }
            break;
        }
        case ExecutorType::DPU: {
            if (type == TaskType::ALL || type == TaskType::DPU_SW) {
                std::vector<DebugInfo> dpuTasks = parseDebugDPUTaskProfiling(dpuTaskList, offset, length, device);
                rawProfTask.insert(rawProfTask.end(), dpuTasks.begin(), dpuTasks.end());
            }
            break;
        }
        case ExecutorType::NONE: {
            VPUX_THROW("None is not a valid profiling executor.");
        }
        }
    }

    return rawProfTask;
}

ClusterTaskArray groupClusterTasks(const RawProfilingRecords& rawTasks) {
    ClusterTaskArray clusterInfoTasks;

    // Grouping of variants into one invariant
    std::multimap<std::pair<std::string, size_t>, RawProfilingRecordPtr> groupedClustersInfo;
    for (const auto &task : rawTasks) {
        const auto clusterId = std::dynamic_pointer_cast<ClusteredAndTiledMixin>(task)->getClusterId();
        const auto key = std::make_pair(task->getRawName(), clusterId);
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
    for (const auto &task : clusterInfoTasks) {
        groupedDPUTasksInfo.insert(std::make_pair(task->getRawName(), task));
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

// At parse time we don't know frequency for some platforms, so data is collected in cycles format. We need to determine
// frequency to convert from cycles to nanoseconds
std::vector<TaskInfo> convertRawTasksToTaskInfo(const RawTasksContainer& rawTasks, const MVCNN::GraphFile* graphFile,
                                                bool fpga, VerbosityLevel verbosity) {
    auto log = vpux::Logger::global();
    const auto device = graphFile->header()->device();

    ClusterTaskArray dpuClusterInfoTasks = groupClusterTasks(rawTasks.dpuTasks);
    RawProfilingRecords dpuInfoTasks = groupTasks(dpuClusterInfoTasks);

    FrequenciesSetup frequenciesSetup;
    bool sameDmaSwCounter = true;
    if (device == MVCNN::TargetDevice::TargetDevice_VPUX37XX) {
        if (fpga) {
            frequenciesSetup = {975.0, 1300.0, 975.0, DMA27Bandwidth};
        } else {
            const auto frequencyEstimationAlgorithm = SynchronizationAlgorithKind::BIDIR_MAX;
            frequenciesSetup = findFrequencySetup37XX(rawTasks.dmaTasks, dpuInfoTasks,
                                                      /*algKind=*/frequencyEstimationAlgorithm);
        }
    } else if (device == MVCNN::TargetDevice::TargetDevice_VPUX311X ||
               device == MVCNN::TargetDevice::TargetDevice_VPUX30XX) {
        sameDmaSwCounter = false;
        frequenciesSetup.profClk = getNceFreq(graphFile);
        frequenciesSetup.dmaBandwidth = DMA20Bandwidth;
    } else {
        VPUX_THROW("Unknown VPUX target device '{0}'", device);
    }

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
        fillTaskInfoWithParsedRawRecords(swTaskInfo, actClusterInfoTasks.getValue(), frequenciesSetup);
    }
    if (verbosity >= VerbosityLevel::HIGH) {
        fillTaskInfoWithParsedRawRecords(dpuTaskInfo, rawTasks.dpuTasks, frequenciesSetup);
        fillTaskInfoWithParsedRawRecords(swTaskInfo, rawTasks.swTasks, frequenciesSetup);
    }

    const auto earliestDmaNs = getEarliestTaskBegin(dmaTaskInfo);
    const auto earliestSwNs = getEarliestTaskBegin(swTaskInfo);

    const auto timersShift = getDMA2OtherTimersShift(rawTasks.dmaTasks, dpuInfoTasks, frequenciesSetup,
                                                     SynchronizationPointKind::DMA_TO_DPU, log);
    log.trace("Timers DMA2DPU difference: {0}", timersShift);
    log.trace("Earliest DMA: {0}", earliestDmaNs);
    log.trace("Earliest DPU: {0}", earliestDpuNs);
    log.trace("Earliest SW : {0}", earliestSwNs);

    if (!dmaTaskInfo.empty()) {
        adjustZeroPoint(dmaTaskInfo, 0, earliestDmaNs.getValue());
    }

    if (!dpuTaskInfo.empty()) {
        const auto dma2dpuOffset = getTimersOffset(timersShift, earliestDmaNs, earliestDpuNs.getValue());
        adjustZeroPoint(dpuTaskInfo, dma2dpuOffset, earliestDmaNs);
        std::sort(dpuTaskInfo.begin(), dpuTaskInfo.end(), [](const TaskInfo& a, const TaskInfo& b) {
            if (a.start_time_ns != b.start_time_ns) {
                return a.start_time_ns < b.start_time_ns;
            } else {
                return std::strcmp(a.name, b.name) < 0;
            }
        });
    }

    int64_t dma2SwOffset = 0;
    if (!swTaskInfo.empty() && !sameDmaSwCounter) {
        const auto dma2UpaTimerShift = getDMA2OtherTimersShift(rawTasks.dmaTasks, rawTasks.swTasks, frequenciesSetup,
                                                               SynchronizationPointKind::DMA_TO_UPA, log);
        log.trace("Timers DMA2UPA difference: {0}", dma2UpaTimerShift);
        dma2SwOffset = getTimersOffset(dma2UpaTimerShift, earliestDmaNs, earliestSwNs.getValue());
    }
    adjustZeroPoint(swTaskInfo, dma2SwOffset, earliestDmaNs);

    std::vector<TaskInfo> allTaskInfo;
    allTaskInfo.reserve(dpuTaskInfo.size() + dmaTaskInfo.size() + swTaskInfo.size());
    allTaskInfo.insert(allTaskInfo.end(), dpuTaskInfo.begin(), dpuTaskInfo.end());
    allTaskInfo.insert(allTaskInfo.end(), dmaTaskInfo.begin(), dmaTaskInfo.end());
    allTaskInfo.insert(allTaskInfo.end(), swTaskInfo.begin(), swTaskInfo.end());

    for (auto& task : allTaskInfo) {
        cleanTaskName(task.name);
    }

    return allTaskInfo;
}

std::vector<TaskInfo> vpux::profiling::getTaskInfo(const uint8_t* blobData, size_t blobSize, const uint8_t* profData,
                                                   size_t profSize, TaskType type, VerbosityLevel verbosity,
                                                   bool fpga) {
    (void)blobSize;
    std::ignore = verbosity;

    if ((nullptr == blobData) || (nullptr == profData)) {
        VPUX_THROW("Empty input data");
    }

    const auto* graphFile = MVCNN::GetGraphFile(blobData);
    // Obtaining FRC speed from blob //
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
    const auto offsets = getProfilingOffsets(graphFile, profSize);

    RawTasksContainer rawTasksContainer;

    for (const auto& p : offsets) {
        const auto offset = p.second.first;
        const auto length = p.second.second;

        switch (p.first) {
        case ExecutorType::DMA: {
            parseDMATaskProfiling(rawTasksContainer.dmaTasks, dmaTaskList, profData + offset, length, device);
            break;
        }
        case ExecutorType::UPA: {
            if (type == TaskType::ALL || type == TaskType::DPU_SW) {
                parseUPATaskProfiling(rawTasksContainer.swTasks, swTaskList, profData + offset, length);
            }
            break;
        }
        case ExecutorType::ACTSHAVE: {
            if (type == TaskType::ALL || type == TaskType::DPU_SW) {
                parseActShaveTaskProfiling(rawTasksContainer.swTasks, swTaskList, profData + offset, length);
            }
            break;
        }
        case ExecutorType::DPU: {
            if (type == TaskType::ALL || type == TaskType::DPU_SW) {
                parseDPUTaskProfiling(rawTasksContainer.dpuTasks, dpuTaskList, profData + offset, length, device);
            }
            break;
        }
        case ExecutorType::NONE: {
            VPUX_THROW("None is not a valid profiling executor.");
        }
        }
    }

    return convertRawTasksToTaskInfo(rawTasksContainer, graphFile, fpga, verbosity);
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
        std::string taskName = std::string(task.name);
        if (taskName.find("cluster_") != std::string::npos) {
            // Skipping high verbose tasks with cluster/variant info
            continue;
        }

        cleanLayerName(taskName);

        auto result = std::find_if(begin(layerInfo), end(layerInfo), [&](LayerInfo item) {
            return taskName == item.name;
        });
        if (result == end(layerInfo)) {
            LayerInfo info = LayerInfo();
            taskName.copy(info.name, sizeof(info.name) - 1);
            info.name[sizeof(info.name) - 1] = '\0';
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
            strncpy(layer->layer_type, task.layer_type, sizeof(LayerInfo::layer_type) - 1);
            layer->layer_type[sizeof(layer->layer_type) - 1] = '\0';
        }
        if (task.exec_type == TaskInfo::ExecType::SW) {
            layer->sw_ns += task.duration_ns;
            strncpy(layer->layer_type, task.layer_type, sizeof(LayerInfo::layer_type) - 1);
            layer->layer_type[sizeof(layer->layer_type) - 1] = '\0';
        }
        if (task.exec_type == TaskInfo::ExecType::DMA) {
            layer->dma_ns += task.duration_ns;
        }
    }

    return layerInfo;
}
