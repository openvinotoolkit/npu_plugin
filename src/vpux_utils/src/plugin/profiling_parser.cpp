//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/utils/plugin/profiling_parser.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/logger.hpp"

#include <map>
#include <sstream>
#include <string>

#include <flatbuffers/flatbuffers.h>
#include <schema/graphfile_generated.h>

using namespace vpux;
using namespace vpux::profiling;

const std::string MULTICLUSTER_PROFILING_SUFFIX = "/_cluster_0";
enum Executor_t {
    NONE,
    DPU,
    UPA,
    ACTSHAVE,
    DMA,
};

typedef struct {
    uint32_t vpuClk;
    uint32_t dpuClk;
    double profClk;
} Timers3720_t;

struct DPUProfilingMeta {
    std::string taskName;
    int32_t taskId;
    int32_t memoryId;
    int32_t maxVariants;
    int32_t numVariants;
    int32_t clusterId;

    uint64_t getOrderDescriptor() const {
        return (memoryId << 16) | clusterId;
    }
};

static Timers3720_t getTimerSpeedMhz3720() {
    // TODO Choose correct freq E#42004
    const Timers3720_t timersFpga{975, 1300, 975};
    // const Timers3720_t timers_3720_highvcc{975, 1300, 38.4};
    // const Timers3720_t timers_3720_midvcc{825, 1100, 38.4};
    // const Timers3720_t timers_3720_lowvcc{525, 700, 38.4};
    return timersFpga;
}

// This class keeps track of closest DMA-(DPU/SW) tasks per layer.
// Required for timers synchronization.
class LayerTimes {
    uint64_t dmaEndNs;
    uint64_t taskStartNs;
    const flatbuffers::Vector<uint32_t>* taskWaitBarriersList;

public:
    LayerTimes(): dmaEndNs(0), taskStartNs(std::numeric_limits<uint64_t>::max()), taskWaitBarriersList(nullptr) {
    }

    bool isValid() const {
        return this->dmaEndNs != 0ul && this->taskStartNs != std::numeric_limits<uint64_t>::max();
    }

    uint64_t getTaskStartNs() const {
        return this->taskStartNs;
    }

    int64_t getDiff() const {
        return (int64_t)(this->dmaEndNs) - (int64_t)(this->taskStartNs);
    }

    void updateStartTimeAndBarriers(const flatbuffers::Vector<uint32_t>* barriersList, uint64_t taskStartNs) {
        if (taskStartNs < this->taskStartNs) {
            this->taskStartNs = taskStartNs;
            this->taskWaitBarriersList = barriersList;
        }
    }

    void updateLayerDmaEndTime(const flatbuffers::Vector<uint32_t>* barriersList, uint64_t taskEndNs) {
        if (this->taskWaitBarriersList == nullptr) {
            return;
        }
        for (const auto barrier : *(this->taskWaitBarriersList)) {
            if (std::find(barriersList->cbegin(), barriersList->cend(), barrier) != barriersList->cend()) {
                this->dmaEndNs = std::max(taskEndNs, this->dmaEndNs);
            }
        }
    };
};

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

static std::map<Executor_t, std::pair<uint32_t, uint32_t>> getProfilingsOffets(const MVCNN::GraphFile* graphFile,
                                                                               size_t profSize) {
    auto profilingOutputs = graphFile->header()->profiling_output();
    VPUX_THROW_UNLESS(profilingOutputs, "Blob contains no profiling_output");
    VPUX_THROW_UNLESS(profilingOutputs->size() == 1, "Blob must contain exactly one profiling output");

    const std::map<std::string, Executor_t> converter{{"dma", Executor_t::DMA},
                                                      {"dpu", Executor_t::DPU},
                                                      {"upa", Executor_t::UPA},
                                                      {"actshave", Executor_t::ACTSHAVE}};

    const char delimiter = '_';
    const std::string outputName = profilingOutputs->Get(0)->name()->str();

    std::stringstream sstream(outputName);
    std::string token;
    std::vector<std::string> tokens;
    while (std::getline(sstream, token, delimiter)) {
        tokens.push_back(token);
    }

    // Store starting offset and size of given profiling data type
    std::map<Executor_t, std::pair<uint32_t, uint32_t>> offsets;
    uint32_t nextOffset = static_cast<uint32_t>(profSize);

    for (auto it = tokens.crbegin(); it != tokens.crend(); ++it) {
        const Executor_t executorEngine = converter.at(*it);
        ++it;
        uint32_t currentOffset = std::stoul(*it);
        offsets[executorEngine] = std::make_pair(currentOffset, static_cast<uint32_t>(nextOffset - currentOffset));
        nextOffset = currentOffset;
    }

    return offsets;
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

static std::vector<TaskInfo> parseDMATaskProfiling(
        const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* dmaTaskList, const void* output, size_t outputLen,
        double frcSpeedMhz, MVCNN::TargetDevice device) {
    if (dmaTaskList == nullptr) {
        return {};
    }

    auto log = Logger::global();
    uint64_t overflowShift = 0;
    uint32_t lastTime = 0;

    std::vector<TaskInfo> profInfo;
    for (unsigned dmaTaskListId = 0; dmaTaskListId < (*dmaTaskList).size(); dmaTaskListId++) {
        auto task = (*dmaTaskList)[dmaTaskListId];
        if ((task->task_as_NNDMATask()->src()->name()->str() == "profilingInput:0") ||
            (task->task_as_NNDMATask()->src()->locale() == MVCNN::MemoryLocation_AbsoluteAddr)) {
            auto taskName = task->name()->str();

            std::string profilingMeta[3];
            getProfilingMeta(taskName, 3, profilingMeta);

            if ((profilingMeta[2] != "PROFTASKBEGIN") && (profilingMeta[2] != "PROFBEGIN")) {
                unsigned layerNumber = 0;
                TaskInfo profInfoItem = TaskInfo();
                profInfoItem.layer_type[0] = '\0';
                profInfoItem.exec_type = TaskInfo::ExecType::DMA;

                layerNumber = stoi(profilingMeta[2]);
                unsigned lastDMAid = stoi(profilingMeta[1]);
                auto currentDMAid = layerNumber * 2 - 1;

                taskName = taskName.substr(0, taskName.find("_PROF"));
                auto typeLen = sizeof(profInfoItem.name) / sizeof(profInfoItem.name[0]);
                auto length = taskName.copy(profInfoItem.name, typeLen, 0);
                profInfoItem.name[length] = '\0';

                if (device != MVCNN::TargetDevice::TargetDevice_VPUX37XX) {
                    if ((currentDMAid >= outputLen / sizeof(uint32_t)) || (lastDMAid >= outputLen / sizeof(uint32_t))) {
                        log.error("Can't process DMA profiling data.");
                        continue;
                    }
                    auto outputBin = reinterpret_cast<const uint32_t*>(output);
                    // Use unsigned 32-bit arithmetic to automatically avoid overflow
                    uint32_t diff = outputBin[currentDMAid] - outputBin[lastDMAid];

                    // Catch overflow and increase overflow shift for absolute start time
                    if (lastTime > 0x7F000000 && outputBin[lastDMAid] < 0x7F000000) {
                        overflowShift += 0x100000000;
                    }
                    lastTime = outputBin[lastDMAid];

                    // Convert to us //
                    profInfoItem.start_time_ns =
                            (uint64_t)(((uint64_t)outputBin[lastDMAid] + overflowShift) * 1000 / frcSpeedMhz);
                    profInfoItem.duration_ns = (uint64_t)((uint64_t)diff * 1000 / frcSpeedMhz);
                } else {
                    if ((currentDMAid >= outputLen / sizeof(uint64_t)) || (lastDMAid >= outputLen / sizeof(uint64_t))) {
                        continue;
                    }
                    auto outputBin = reinterpret_cast<const uint64_t*>(output);
                    uint64_t diff = outputBin[currentDMAid] - outputBin[lastDMAid];
                    // Convert to us //
                    profInfoItem.start_time_ns = (uint64_t)(outputBin[lastDMAid] * 1000 / frcSpeedMhz);
                    profInfoItem.duration_ns = (uint64_t)(diff * 1000 / frcSpeedMhz);
                }
                profInfoItem.task_id = dmaTaskListId;

                profInfo.push_back(profInfoItem);
            }
        }
    }
    return profInfo;
}

static std::vector<TaskInfo> parseUPATaskProfiling(
        const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* upaTaskList, const void* output, size_t outputLen,
        double frcSpeedMhz) {
    struct UpaData_t {
        uint64_t begin;
        uint64_t end;
        uint32_t stallCycles;
        uint32_t activeCycles;
    };

    if (upaTaskList == nullptr) {
        return {};
    }

    auto log = Logger::global();
    auto outputUpa = reinterpret_cast<const UpaData_t*>(output);

    std::vector<TaskInfo> profInfo;
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

            if (currentPos >= outputLen / sizeof(UpaData_t) ||
                (outputUpa[currentPos].begin == 0 && outputUpa[currentPos].end == 0)) {
                log.error("Can't process UPA profiling data.");
                continue;
            }

            TaskInfo profInfoItem = TaskInfo();
            auto softLayer = task->task_as_UPALayerTask();
            if (softLayer != nullptr) {
                const auto typeLen = sizeof(profInfoItem.layer_type);
                const char* typeName = EnumNameSoftwareLayerParams(softLayer->softLayerParams_type());
                if (typeName != nullptr) {
                    strncpy(profInfoItem.layer_type, typeName, typeLen - 1);
                }
            } else {
                profInfoItem.layer_type[0] = '\0';
            }
            profInfoItem.exec_type = TaskInfo::ExecType::SW;
            uint64_t diff = outputUpa[currentPos].end - outputUpa[currentPos].begin;
            profInfoItem.start_time_ns = (uint64_t)(outputUpa[currentPos].begin * 1000 / frcSpeedMhz);
            profInfoItem.duration_ns = (uint64_t)(diff * 1000 / frcSpeedMhz);
            profInfoItem.active_cycles = outputUpa[currentPos].activeCycles;
            profInfoItem.stall_cycles = outputUpa[currentPos].stallCycles;
            profInfoItem.task_id = upaTaskListId;

            const auto nameLen = sizeof(profInfoItem.name) / sizeof(profInfoItem.name[0]);
            const auto length = taskName.copy(profInfoItem.name, nameLen, 0);
            profInfoItem.name[length] = '\0';
            profInfo.push_back(profInfoItem);
        }
    }
    return profInfo;
}

uint64_t convertTicksToNs(uint64_t ticks, double frcSpeedMhz) {
    return static_cast<uint64_t>(ticks * 1000 / frcSpeedMhz);
}

uint64_t noOverflowSubtract(uint64_t first, uint64_t second, uint64_t max) {
    return first - second + ((first < second) ? max : 0);
}
std::string getPrefix(const std::string& str, const std::string& delimeter) {
    return str.substr(0, str.find(delimeter));
}

DPUProfilingMeta parse(const std::string& fullTaskName) {
    const int MIN_NUMBER_OF_SEGMENTS = 4;
    const int NUMBER_OF_SEGMENTS_WITH_CLUSTER_INFO = 6;
    const int MAX_NUMBER_OF_SEGMENTS = NUMBER_OF_SEGMENTS_WITH_CLUSTER_INFO;

    DPUProfilingMeta meta;
    auto size = MIN_NUMBER_OF_SEGMENTS;
    int32_t currentClusterId = 0;
    if (fullTaskName.find("_cluster_") != fullTaskName.npos) {
        size = NUMBER_OF_SEGMENTS_WITH_CLUSTER_INFO;
    }
    std::string segments[MAX_NUMBER_OF_SEGMENTS];
    getProfilingMeta(fullTaskName, size, segments);
    if (size == NUMBER_OF_SEGMENTS_WITH_CLUSTER_INFO) {
        currentClusterId = std::stoi(segments[5]);
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

    int32_t clustersAmount = std::stoi(segments[2]);
    const auto dpuTasksDistribution = segments[3];
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
    meta.clusterId = currentClusterId;
    meta.maxVariants = std::stoi(maxVariants);
    meta.numVariants = numVariants;
    return meta;
}

static std::vector<TaskInfo> parseActShaveTaskProfiling(
        const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* shaveTaskList, const void* output,
        size_t outputLen, double frcSpeedMhz) {
    struct ActShaveData_t {
        uint64_t begin;
        uint32_t duration;
        uint32_t stallCycles;
    };

    if (shaveTaskList == nullptr) {
        return {};
    }

    auto log = Logger::global();
    const ActShaveData_t* outputShave = reinterpret_cast<const ActShaveData_t*>(output);
    const size_t numOfActShaveTasks = outputLen / sizeof(ActShaveData_t);

    std::vector<TaskInfo> profInfo;
    for (const auto& task : *shaveTaskList) {
        auto taskName = task->name()->str();
        std::string profilingMeta[2];
        getProfilingMeta(taskName, 2, profilingMeta);

        if (profilingMeta[0] == "PROF") {
            taskName = taskName.substr(0, taskName.find("_PROF"));
            if (!taskName.empty() && taskName[taskName.length() - 1] == '/') {
                taskName.pop_back();
            }
            unsigned currentPos = stoi(profilingMeta[1]);

            if (currentPos >= numOfActShaveTasks ||
                (outputShave[currentPos].begin == 0 && outputShave[currentPos].duration == 0)) {
                log.error("Can't process ActShave profiling data.");
                continue;
            }

            TaskInfo profInfoItem = TaskInfo();
            auto softLayer = task->task_as_ActKernelTask();
            if (softLayer != nullptr) {
                // auto typeLen = sizeof(profInfoItem.layer_type) / sizeof(profInfoItem.layer_type[0]);
                // strncpy(profInfoItem.layer_type, EnumNameSoftwareLayerParams(softLayer->softLayerParams_type()),
                //         typeLen - 1);
            } else {
                profInfoItem.layer_type[0] = '\0';
            }
            profInfoItem.exec_type = TaskInfo::ExecType::SW;
            profInfoItem.start_time_ns = convertTicksToNs(outputShave[currentPos].begin, frcSpeedMhz);
            profInfoItem.duration_ns = convertTicksToNs(outputShave[currentPos].duration, frcSpeedMhz);
            profInfoItem.active_cycles = 0;
            profInfoItem.stall_cycles = outputShave[currentPos].stallCycles;
            profInfoItem.task_id = currentPos;

            auto typeLen = sizeof(profInfoItem.name) / sizeof(profInfoItem.name[0]);
            auto length = taskName.copy(profInfoItem.name, typeLen, 0);
            profInfoItem.name[length] = '\0';
            profInfo.push_back(profInfoItem);
        }
    }
    return profInfo;
}

static std::pair<uint64_t, uint64_t> getStartEndTimestampsSw(const void* profBuffer, size_t profSize, size_t index,
                                                             double frcSpeedMhz) {
    // KMB DPU profiling data payload
    struct DpuData_t {
        uint64_t begin;
        uint64_t end;
    };

    const DpuData_t* outputDpu = reinterpret_cast<const DpuData_t*>(profBuffer);
    VPUX_THROW_WHEN(index >= profSize / sizeof(DpuData_t) || (outputDpu[index].begin == 0 && outputDpu[index].end == 0),
                    "Unexpected end of blob or empty DPU profiling data");

    const uint64_t taskBeginCandidate = convertTicksToNs(outputDpu[index].begin, frcSpeedMhz);
    const uint64_t taskEndCandidate = convertTicksToNs(outputDpu[index].end, frcSpeedMhz);

    return {taskBeginCandidate, taskEndCandidate};
}

static std::pair<uint64_t, uint64_t> getStartEndTimestampsHwp(const void* profBuffer, size_t profSize, size_t index) {
    // DPU HWP profiling data payload
    struct HpwDpuMode0Data_t {
        uint64_t idu_wl_duration : 28;
        uint64_t idu_tstamp : 28;
        uint64_t sve_id : 5;
        uint64_t : 3;
        uint64_t odu_wl_duration : 28;
        uint64_t odu_tstamp : 28;
        uint64_t : 8;
    };

    const auto timers = getTimerSpeedMhz3720();
    const uint32_t vpuClk = timers.vpuClk;
    const uint32_t dpuClk = timers.dpuClk;
    const uint64_t max28bitVal = convertTicksToNs(0x0FFFFFFFull, vpuClk);

    const HpwDpuMode0Data_t* outputDpu = reinterpret_cast<const HpwDpuMode0Data_t*>(profBuffer);
    VPUX_THROW_WHEN(index >= profSize / sizeof(HpwDpuMode0Data_t) ||
                            (outputDpu[index].idu_wl_duration == 0 && outputDpu[index].odu_wl_duration == 0),
                    "Unexpected end of blob or empty DPU profiling data");

    const uint64_t taskBeginCandidate =
            noOverflowSubtract(convertTicksToNs(outputDpu[index].idu_tstamp, vpuClk),
                               convertTicksToNs(outputDpu[index].idu_wl_duration, dpuClk), max28bitVal);
    const uint64_t taskEndCandidate = convertTicksToNs(outputDpu[index].odu_tstamp, vpuClk);

    return {taskBeginCandidate, taskEndCandidate};
}

void setProfInfoName(TaskInfo& profInfoItem, const std::string& taskName) {
    auto typeLen = sizeof(profInfoItem.name) / sizeof(profInfoItem.name[0]);
    auto length = taskName.copy(profInfoItem.name, typeLen, 0);
    profInfoItem.name[length] = '\0';
}

static std::vector<TaskInfo> parseDPUTaskProfiling(
        const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* dpuTaskList, const void* output, size_t outputLen,
        double frcSpeedMhz, MVCNN::TargetDevice device, std::vector<TaskInfo>& nceTaskProfilingInfo,
        std::vector<TaskInfo>& variantProfilingInfo) {
    if (dpuTaskList == nullptr) {
        return {};
    }

    // Ordering profiling tasks in ascending order
    std::map<uint64_t, DPUProfilingMeta> profilingMeta;
    for (unsigned dpu_taskListId = 0; dpu_taskListId < (*dpuTaskList).size(); dpu_taskListId++) {
        auto task = (*dpuTaskList)[dpu_taskListId];
        auto taskName = task->name()->str();
        auto meta = parse(taskName);
        meta.taskId = dpu_taskListId;
        profilingMeta.emplace(meta.getOrderDescriptor(), meta);
    }

    unsigned currentPos = 0;
    int32_t expectedMemoryId = -1;
    std::vector<TaskInfo> profInfo;
    for (const auto& iter : profilingMeta) {
        const auto metaData = iter.second;

        if (metaData.clusterId == 0) {
            ++expectedMemoryId;
        }
        VPUX_THROW_WHEN(expectedMemoryId != metaData.memoryId,
                        "Profiling tasks inside buffer should be located sequentially");

        const auto clusterName =
                metaData.taskName + CLUSTER_LEVEL_PROFILING_SUFFIX + std::to_string(metaData.clusterId);

        TaskInfo profInfoItem = TaskInfo();
        profInfoItem.layer_type[0] = '\0';
        profInfoItem.task_id = metaData.taskId;
        profInfoItem.exec_type = TaskInfo::ExecType::DPU;
        profInfoItem.active_cycles = 0;
        profInfoItem.stall_cycles = 0;

        uint64_t clusterStartTimeNs = std::numeric_limits<uint64_t>::max();
        uint64_t clusterFinishTimeNs = std::numeric_limits<uint64_t>::min();
        for (auto variantId = 0; variantId < metaData.maxVariants; variantId++) {
            if (variantId < metaData.numVariants) {
                std::pair<uint64_t, uint64_t> timestampCandidates;

                if (device == MVCNN::TargetDevice::TargetDevice_VPUX37XX) {
                    timestampCandidates = getStartEndTimestampsHwp(output, outputLen, currentPos);
                } else {
                    timestampCandidates = getStartEndTimestampsSw(output, outputLen, currentPos, frcSpeedMhz);
                }

                profInfoItem.start_time_ns = timestampCandidates.first;
                profInfoItem.duration_ns = timestampCandidates.second - timestampCandidates.first;

                clusterStartTimeNs = std::min(clusterStartTimeNs, timestampCandidates.first);
                clusterFinishTimeNs = std::max(clusterFinishTimeNs, timestampCandidates.second);

                const auto variantName = clusterName + VARIANT_LEVEL_PROFILING_SUFFIX + std::to_string(variantId);
                setProfInfoName(profInfoItem, variantName);
                variantProfilingInfo.push_back(profInfoItem);
            }
            // continue increament of currentPos to walk over non-used data
            ++currentPos;
        }
        // Since variant/cluster/task profiling info have a lot of common fields, reuse them and just change
        // name/timestamps
        setProfInfoName(profInfoItem, clusterName);
        profInfoItem.start_time_ns = clusterStartTimeNs;
        profInfoItem.duration_ns = clusterFinishTimeNs - clusterStartTimeNs;
        nceTaskProfilingInfo.push_back(profInfoItem);

        // profInfo storing information at highest abstraction level(Task Level), which combine information from all
        // clusters and underlying variants. Tasks are located sequentially(quaranteed by map usage) in order
        // task-X/cluster_0, task-X/cluster_1.... and each cluster shares information about name and wait
        // barriers(taskId reffer to them) so possible just to update timestamps in last item from profInfo
        if (profInfo.empty() || profInfo.back().exec_type != TaskInfo::ExecType::DPU ||
            metaData.taskName != profInfo.back().name) {
            setProfInfoName(profInfoItem, metaData.taskName);
            profInfo.push_back(profInfoItem);
        }
        TaskInfo& currentTask = profInfo.back();
        const auto oldFinishTimeNs = currentTask.start_time_ns + currentTask.duration_ns;

        currentTask.start_time_ns = std::min(currentTask.start_time_ns, profInfoItem.start_time_ns);
        currentTask.duration_ns = std::max(oldFinishTimeNs, clusterFinishTimeNs) - currentTask.start_time_ns;
    }
    return profInfo;
}

static std::map<std::string, LayerTimes> collectEarliestTasksPerLayer(
        const std::vector<TaskInfo>& taskInfo,
        const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* dpuTaskList) {
    std::map<std::string, LayerTimes> layerInfoTimes;

    for (const auto& task : taskInfo) {
        const std::string name = std::string(task.name);
        LayerTimes& layer = layerInfoTimes[name];  // Create or insert new
        if (dpuTaskList != nullptr) {
            layer.updateStartTimeAndBarriers(dpuTaskList->Get(task.task_id)->associated_barriers()->wait_barriers(),
                                             task.start_time_ns);
        }
    }
    return layerInfoTimes;
}

// Lets find the minimal offset between timers as we know for sure that DPU/SW task are started after the end of
// DMA task because they are connected via the same barrier
static int64_t getTimersOffset(const std::map<std::string, LayerTimes>& layers, uint64_t earliestDmaNs,
                               uint64_t earliestTaskNs) {
    int64_t taskTimerDiff = std::numeric_limits<int64_t>::min();
    for (auto& times : layers) {
        const LayerTimes& layer = times.second;
        if (layer.isValid()) {
            taskTimerDiff = std::max(taskTimerDiff, layer.getDiff());
        }
    }

    if (taskTimerDiff == std::numeric_limits<int64_t>::min()) {
        // Could not calculate offset between timers(Most likely DMA profiling is disabled)
        // -> set offset based on begin time
        if (earliestDmaNs != std::numeric_limits<uint64_t>::max()) {
            taskTimerDiff = earliestDmaNs - earliestTaskNs;
        } else {
            taskTimerDiff = -earliestTaskNs;
        }
    }

    return taskTimerDiff;
};

// Adjust all tasks to zero point (earliest DMA task)
static void adjustZeroPoint(std::vector<vpux::profiling::TaskInfo>& taskInfo, int64_t timerDiff, int64_t firstDma) {
    for (auto& task : taskInfo) {
        int64_t startTimeNs = task.start_time_ns - firstDma + timerDiff;
        task.start_time_ns = std::max(startTimeNs, (int64_t)0);
    }
};

std::vector<TaskInfo> vpux::profiling::getTaskInfo(const uint8_t* blobData, size_t blobSize, const uint8_t* profData,
                                                   size_t profSize, TaskType type, VerbosityLevel verbosity) {
    (void)blobSize;

    if ((nullptr == blobData) || (nullptr == profData)) {
        VPUX_THROW("Empty input data");
    }

    const auto* graphFile = MVCNN::GetGraphFile(blobData);
    // Obtaining FRC speed from blob //
    const double frcSpeedMhz = getNceFreq(graphFile);

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
    const auto offsets = getProfilingsOffets(graphFile, profSize);
    const auto outputBytes = reinterpret_cast<const uint8_t*>(profData);

    std::vector<TaskInfo> nceTaskProfilingInfo;
    std::vector<TaskInfo> variantProfilingInfo;

    std::vector<TaskInfo> dmaTaskInfo;
    std::vector<TaskInfo> dpuTaskInfo;
    std::vector<TaskInfo> swTaskInfo;

    for (const auto& p : offsets) {
        const auto offset = p.second.first;
        const auto length = p.second.second;

        switch (p.first) {
        case Executor_t::DMA: {
            if (type == TaskType::ALL || type == TaskType::DMA) {
                dmaTaskInfo = parseDMATaskProfiling(dmaTaskList, outputBytes + offset, length, frcSpeedMhz,
                                                    graphFile->header()->device());
            }
            break;
        }
        case Executor_t::UPA: {
            if (type == TaskType::ALL || type == TaskType::DPU_SW) {
                swTaskInfo = parseUPATaskProfiling(swTaskList, outputBytes + offset, length, frcSpeedMhz);
            }
            break;
        }
        case Executor_t::ACTSHAVE: {
            if (type == TaskType::ALL || type == TaskType::DPU_SW) {
                swTaskInfo = parseActShaveTaskProfiling(swTaskList, outputBytes + offset, length,
                                                        getTimerSpeedMhz3720().profClk);
            }
            break;
        }
        case Executor_t::DPU: {
            if (type == TaskType::ALL || type == TaskType::DPU_SW) {
                dpuTaskInfo = parseDPUTaskProfiling(dpuTaskList, outputBytes + offset, length, frcSpeedMhz,
                                                    graphFile->header()->device(), nceTaskProfilingInfo,
                                                    variantProfilingInfo);
            }
            break;
        }
        case Executor_t::NONE: {
            VPUX_THROW("None is not a valid profiling executor.");
        }
        }
    }

    std::map<std::string, LayerTimes> dpuLayerInfoTimes = collectEarliestTasksPerLayer(dpuTaskInfo, dpuTaskList);
    std::map<std::string, LayerTimes> swLayerInfoTimes = collectEarliestTasksPerLayer(swTaskInfo, swTaskList);

    auto minStartTime = [](const std::pair<std::string, LayerTimes>& a, const std::pair<std::string, LayerTimes>& b) {
        return a.second.getTaskStartNs() < b.second.getTaskStartNs();
    };

    // Find the earliest task of each type
    const auto earliestDpuIter = std::min_element(dpuLayerInfoTimes.cbegin(), dpuLayerInfoTimes.cend(), minStartTime);
    const uint64_t earliestDpuNs = (earliestDpuIter != dpuLayerInfoTimes.cend())
                                           ? earliestDpuIter->second.getTaskStartNs()
                                           : std::numeric_limits<uint64_t>::max();

    const auto earliestSwIter = std::min_element(swLayerInfoTimes.cbegin(), swLayerInfoTimes.cend(), minStartTime);
    const uint64_t earliestSwNs = (earliestSwIter != swLayerInfoTimes.cend()) ? earliestSwIter->second.getTaskStartNs()
                                                                              : std::numeric_limits<uint64_t>::max();

    const auto earliestDmaIter =
            std::min_element(dmaTaskInfo.cbegin(), dmaTaskInfo.cend(), [](const TaskInfo& a, const TaskInfo& b) {
                return a.start_time_ns < b.start_time_ns;
            });
    const uint64_t earliestDmaNs = (earliestDmaIter != dmaTaskInfo.cend()) ? earliestDmaIter->start_time_ns
                                                                           : std::numeric_limits<uint64_t>::max();

    if (dmaTaskList != nullptr) {
        for (const auto& task : dmaTaskInfo) {
            // Finding DMA to DPU/SW timers synchronisation points.
            // DMA task should update and DPU task should wait for the same barrier within one layer
            const auto barriersList = dmaTaskList->Get(task.task_id)->associated_barriers()->update_barriers();
            if (barriersList == nullptr) {
                // This task doesn't have barriers so it is impossible to find its relation to other tasks
                continue;
            }

            const auto name = std::string(task.name);
            const uint64_t taskEndNs = task.start_time_ns + task.duration_ns;

            for (auto layerInfoTimes : {&swLayerInfoTimes, &dpuLayerInfoTimes}) {
                if (layerInfoTimes->count(name) == 1) {
                    layerInfoTimes->at(name).updateLayerDmaEndTime(barriersList, taskEndNs);
                }
            }
        }
    }

    // Get timers difference. This is the cornerstone of timers synchronization.
    const int64_t dmaToDpuTaskTimerDiff = getTimersOffset(dpuLayerInfoTimes, earliestDmaNs, earliestDpuNs);
    const int64_t dmaToSwTaskTimerDiff = getTimersOffset(swLayerInfoTimes, earliestDmaNs, earliestSwNs);

    adjustZeroPoint(dmaTaskInfo, 0, earliestDmaNs);
    adjustZeroPoint(dpuTaskInfo, dmaToDpuTaskTimerDiff, earliestDmaNs);
    adjustZeroPoint(swTaskInfo, dmaToSwTaskTimerDiff, earliestDmaNs);

    // Merge all three containers with tasks
    dpuTaskInfo.insert(dpuTaskInfo.end(), dmaTaskInfo.begin(), dmaTaskInfo.end());
    dpuTaskInfo.insert(dpuTaskInfo.end(), swTaskInfo.begin(), swTaskInfo.end());
    // Merge high-verbosity info if needed
    if (verbosity >= VerbosityLevel::MEDIUM) {
        adjustZeroPoint(nceTaskProfilingInfo, dmaToDpuTaskTimerDiff, earliestDmaNs);
        dpuTaskInfo.insert(dpuTaskInfo.end(), nceTaskProfilingInfo.begin(), nceTaskProfilingInfo.end());
    }
    if (verbosity >= VerbosityLevel::HIGH) {
        adjustZeroPoint(variantProfilingInfo, dmaToDpuTaskTimerDiff, earliestDmaNs);
        dpuTaskInfo.insert(dpuTaskInfo.end(), variantProfilingInfo.begin(), variantProfilingInfo.end());
    }

    // Return all types of tasks despite misleading name
    return dpuTaskInfo;
}

std::vector<LayerInfo> vpux::profiling::getLayerInfo(const uint8_t* blobData, size_t blobSize, const uint8_t* profData,
                                                     size_t profSize) {
    std::vector<TaskInfo> taskInfo =
            getTaskInfo(blobData, blobSize, profData, profSize, TaskType::ALL, VerbosityLevel::LOW);

    return getLayerInfo(taskInfo);
}

std::vector<LayerInfo> vpux::profiling::getLayerInfo(const std::vector<TaskInfo>& taskInfo) {
    std::vector<LayerInfo> layerInfo;
    for (auto& task : taskInfo) {
        LayerInfo* layer;
        std::string taskName = std::string(task.name);
        if (taskName.find("cluster_") != std::string::npos) {
            // Skipping high verbose tasks with cluster/variant info
            continue;
        }
        const auto outputPos = taskName.rfind("/output tile");
        const auto inputPos = taskName.rfind("/input");
        const auto tilePos = taskName.rfind("tile [");
        if (outputPos != std::string::npos) {
            taskName.erase(outputPos);
        } else if (inputPos != std::string::npos && tilePos != std::string::npos) {
            taskName.erase(inputPos);
        }

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
            strncpy(layer->layer_type, task.layer_type, sizeof(LayerInfo::layer_type));
            layer->layer_type[sizeof(layer->layer_type) - 1] = '\0';
        }
        if (task.exec_type == TaskInfo::ExecType::SW) {
            layer->sw_ns += task.duration_ns;
            strncpy(layer->layer_type, task.layer_type, sizeof(LayerInfo::layer_type));
            layer->layer_type[sizeof(layer->layer_type) - 1] = '\0';
        }
        if (task.exec_type == TaskInfo::ExecType::DMA) {
            layer->dma_ns += task.duration_ns;
        }
    }

    return layerInfo;
}
