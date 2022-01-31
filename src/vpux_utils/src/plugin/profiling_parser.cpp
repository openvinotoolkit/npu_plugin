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

#include "vpux/utils/plugin/profiling_parser.hpp"
#include "vpux/utils/core/error.hpp"

#include <map>

#include <flatbuffers/flatbuffers.h>
#include <schema/graphfile_generated.h>

using namespace vpux;
using namespace vpux::profiling;

typedef struct {
    std::string layer_type;
    std::string hw_type;
} execution_info_t;

static double get_frc_speed(const MVCNN::GraphFile* graphFile) {
    double frc_speed_mhz = 0;
    auto processor_frequencies = graphFile->header()->resources()->processor_frequencies();
    VPUX_THROW_UNLESS(processor_frequencies, "Blob contains no processor_frequencies");
    for (auto frequency : *processor_frequencies) {
        if (frequency->item() == MVCNN::PhysicalProcessor_NCE_Cluster) {
            frc_speed_mhz = frequency->number();
            break;
        }
    }

    if (!frc_speed_mhz) {
        switch (graphFile->header()->device()) {
        case MVCNN::TargetDevice::TargetDevice_KMB:
            switch (graphFile->header()->device_revision()) {
            case MVCNN::TargetDeviceRevision::TargetDeviceRevision_A0:
                frc_speed_mhz = 500;
                break;
            case MVCNN::TargetDeviceRevision::TargetDeviceRevision_B0:
                frc_speed_mhz = 700;
                break;
            default:
                VPUX_THROW("TargetDeviceRevision {0} is not supported",
                           EnumNameTargetDeviceRevision(graphFile->header()->device_revision()));
            }
            break;
        case MVCNN::TargetDevice::TargetDevice_TBH:
            frc_speed_mhz = 700;
            break;
        default:
            VPUX_THROW("TargetDevice {0} is not supported ", EnumNameTargetDevice(graphFile->header()->device()));
        }
    }
    if (!frc_speed_mhz)
        frc_speed_mhz = 500;

    return frc_speed_mhz;
}

static const std::vector<std::pair<TaskInfo::ExecType, uint32_t>> get_profilings_offets(
        const MVCNN::GraphFile* graphFile) {
    std::vector<std::pair<TaskInfo::ExecType, uint32_t>> offsets;

    auto profiling_outputs = graphFile->header()->profiling_output();
    VPUX_THROW_UNLESS(profiling_outputs, "Blob contains no profiling_output");

    for (auto output : *profiling_outputs) {
        const std::string output_name = output->name()->str();
        size_t bpos = 0;
        while (bpos < output_name.length()) {
            size_t pos = output_name.find('_', bpos);
            VPUX_THROW_UNLESS(pos != bpos, "Failed to parse profiling output name");
            auto offset = atoi(output_name.substr(bpos, pos - bpos).c_str());
            bpos = pos + 1;
            pos = output_name.find('_', bpos);
            if (pos == std::string::npos) {
                pos = output_name.length();
            }
            auto name = output_name.substr(bpos, pos - bpos);
            bpos = pos + 1;

            auto type = TaskInfo::ExecType::NONE;
            if (name == "dma") {
                type = TaskInfo::ExecType::DMA;
            } else if (name == "dpu") {
                type = TaskInfo::ExecType::DPU;
            } else if (name == "upa") {
                type = TaskInfo::ExecType::SW;
            }
            offsets.push_back({type, offset});
        }
    }

    return offsets;
}

void getProfilingMeta(std::string& taskName, unsigned size, std::string* profilingMeta) {
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

static void parseDMATaskProfiling(const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* dma_taskList,
                                  const void* output, size_t output_len, double frc_speed_mhz,
                                  std::vector<TaskInfo>& profInfo) {
    if (dma_taskList == nullptr) {
        return;
    }
    auto output_bin = reinterpret_cast<const uint32_t*>(output);
    uint64_t overflow_shift = 0;
    uint32_t last_time = 0;

    for (unsigned dma_taskListId = 0; dma_taskListId < (*dma_taskList).size(); dma_taskListId++) {
        auto task = (*dma_taskList)[dma_taskListId];
        if ((task->task_as_NNDMATask()->src()->name()->str() == "profilingInput:0") ||
            (task->task_as_NNDMATask()->src()->locale() == MVCNN::MemoryLocation_AbsoluteAddr)) {
            auto taskName = task->name()->str();

            std::string profiling_meta[3];
            getProfilingMeta(taskName, 3, profiling_meta);

            if ((profiling_meta[2] != "PROFTASKBEGIN") && (profiling_meta[2] != "PROFBEGIN")) {
                unsigned layerNumber = 0;
                TaskInfo profInfoItem = TaskInfo();
                profInfoItem.layer_type[0] = '\0';
                profInfoItem.exec_type = TaskInfo::ExecType::DMA;

                layerNumber = stoi(profiling_meta[2]);
                unsigned lastDMAid = stoi(profiling_meta[1]);
                auto currentDMAid = layerNumber * 2 - 1;

                if ((currentDMAid >= output_len / sizeof(uint32_t)) || (lastDMAid >= output_len / sizeof(uint32_t))) {
                    continue;
                }
                // Use unsigned 32-bit arithmetic to automatically avoid overflow
                uint32_t diff = output_bin[currentDMAid] - output_bin[lastDMAid];
                // Catch otherflow and increase otherflow shift for absolute start time
                if (last_time > 0x7F000000 && output_bin[lastDMAid] < 0x7F000000) {
                    overflow_shift += 0x100000000;
                }
                last_time = output_bin[lastDMAid];

                taskName = taskName.substr(0, taskName.find("_PROF"));
                auto typeLen = sizeof(profInfoItem.name) / sizeof(profInfoItem.name[0]);
                auto length = taskName.copy(profInfoItem.name, typeLen, 0);
                profInfoItem.name[length] = '\0';
                // Convert to us //
                profInfoItem.start_time_ns =
                        (uint64_t)(((uint64_t)output_bin[lastDMAid] + overflow_shift) * 1000 / frc_speed_mhz);
                profInfoItem.duration_ns = (uint64_t)((uint64_t)diff * 1000 / frc_speed_mhz);
                profInfoItem.task_id = dma_taskListId;

                profInfo.push_back(profInfoItem);
            }
        }
    }
}

static void parseUPATaskProfiling(const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* upa_taskList,
                                  const void* output, size_t output_len, double frc_speed_mhz,
                                  std::vector<TaskInfo>& profInfo) {
    struct upa_data_t {
        uint64_t begin;
        uint64_t end;
        uint32_t stall_cycles;
        uint32_t active_cycles;
    };

    if (upa_taskList == nullptr) {
        return;
    }
    auto output_upa = reinterpret_cast<const upa_data_t*>(output);

    for (unsigned upa_taskListId = 0; upa_taskListId < (*upa_taskList).size(); upa_taskListId++) {
        auto task = (*upa_taskList)[upa_taskListId];
        auto taskName = task->name()->str();
        std::string profiling_meta[2];
        getProfilingMeta(taskName, 2, profiling_meta);

        if (profiling_meta[0] == "PROF") {
            taskName = taskName.substr(0, taskName.find("_PROF"));
            if (!taskName.empty() && taskName[taskName.length() - 1] == '/') {
                taskName.pop_back();
            }
            unsigned currentPos = stoi(profiling_meta[1]);

            if (currentPos >= output_len / sizeof(upa_data_t) ||
                (output_upa[currentPos].begin == 0 && output_upa[currentPos].end == 0)) {
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
            uint64_t diff = output_upa[currentPos].end - output_upa[currentPos].begin;
            profInfoItem.start_time_ns = (uint64_t)(output_upa[currentPos].begin * 1000 / frc_speed_mhz);
            profInfoItem.duration_ns = (uint64_t)(diff * 1000 / frc_speed_mhz);
            profInfoItem.active_cycles = output_upa[currentPos].active_cycles;
            profInfoItem.stall_cycles = output_upa[currentPos].stall_cycles;
            profInfoItem.task_id = upa_taskListId;

            const auto nameLen = sizeof(profInfoItem.name) / sizeof(profInfoItem.name[0]);
            const auto length = taskName.copy(profInfoItem.name, nameLen, 0);
            profInfoItem.name[length] = '\0';
            profInfo.push_back(profInfoItem);
        }
    }
}

static void parseDPUTaskProfiling(const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* dpu_taskList,
                                  const void* output, size_t output_len, double frc_speed_mhz,
                                  std::vector<TaskInfo>& profInfo) {
    struct dpu_data_t {
        uint64_t begin;
        uint64_t end;
    };

    if (dpu_taskList == nullptr) {
        return;
    }
    auto output_dpu = reinterpret_cast<const dpu_data_t*>(output);

    for (unsigned dpu_taskListId = 0; dpu_taskListId < (*dpu_taskList).size(); dpu_taskListId++) {
        auto task = (*dpu_taskList)[dpu_taskListId];
        auto taskName = task->name()->str();
        std::string profiling_meta[2];
        getProfilingMeta(taskName, 2, profiling_meta);

        if (profiling_meta[0] == "PROF") {
            taskName = taskName.substr(0, taskName.find("_PROF"));
            if (!taskName.empty() && taskName[taskName.length() - 1] == '/') {
                taskName.pop_back();
            }
            unsigned currentPos = stoi(profiling_meta[1]);

            if (currentPos >= output_len / sizeof(dpu_data_t) ||
                (output_dpu[currentPos].begin == 0 && output_dpu[currentPos].end == 0)) {
                continue;
            }

            TaskInfo profInfoItem = TaskInfo();
            profInfoItem.layer_type[0] = '\0';
            profInfoItem.exec_type = TaskInfo::ExecType::DPU;
            uint64_t diff = output_dpu[currentPos].end - output_dpu[currentPos].begin;
            profInfoItem.start_time_ns = (uint64_t)(output_dpu[currentPos].begin * 1000 / frc_speed_mhz);
            profInfoItem.duration_ns = (uint64_t)(diff * 1000 / frc_speed_mhz);
            profInfoItem.active_cycles = 0;
            profInfoItem.stall_cycles = 0;
            profInfoItem.task_id = dpu_taskListId;

            auto typeLen = sizeof(profInfoItem.name) / sizeof(profInfoItem.name[0]);
            auto length = taskName.copy(profInfoItem.name, typeLen, 0);
            profInfoItem.name[length] = '\0';
            profInfo.push_back(profInfoItem);
        }
    }
}

void vpux::profiling::getTaskInfo(const void* data, size_t data_len, const void* output, size_t output_len,
                                  std::vector<TaskInfo>& taskInfo, TaskType type) {
    (void)data_len;

    if ((nullptr == data) || (nullptr == output)) {
        VPUX_THROW("Empty input data");
    }

    const auto* graphFile = MVCNN::GetGraphFile(data);
    // Obtaining FRC speed from blob //
    auto frc_speed_mhz = get_frc_speed(graphFile);

    // Finding of corresponding task list //
    const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* dma_taskList = nullptr;
    const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* dpu_taskList = nullptr;
    const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* upa_taskList = nullptr;
    auto task_lists = graphFile->task_lists();
    VPUX_THROW_UNLESS(task_lists, "Blob contains no task_lists");
    for (auto task_list_item : *task_lists) {
        auto task0_type = task_list_item->content()->Get(0)->task_type();
        if (task0_type == MVCNN::SpecificTask_NNDMATask) {
            dma_taskList = task_list_item->content();
        }
        if (task0_type == MVCNN::SpecificTask_NCE2Task) {
            dpu_taskList = task_list_item->content();
        }
        if (task0_type == MVCNN::SpecificTask_UPALayerTask) {
            upa_taskList = task_list_item->content();
        }
    }

    // Finding offsets of different profiling type in the profiling output //
    const auto offsets = get_profilings_offets(graphFile);
    for (size_t i = 0; i < offsets.size(); i++) {
        auto offset = offsets[i];
        size_t len;
        if (i < offsets.size() - 1) {
            len = offsets[i + 1].second - offset.second;
        } else {
            len = output_len - offset.second;
        }

        if (offset.first == TaskInfo::ExecType::DMA && (type == TaskType::ALL || type == TaskType::DMA)) {
            auto output_bytes = reinterpret_cast<const uint8_t*>(output);
            parseDMATaskProfiling(dma_taskList, output_bytes + offset.second, len, frc_speed_mhz, taskInfo);
        }
        if (offset.first == TaskInfo::ExecType::SW && (type == TaskType::ALL || type == TaskType::DPU_SW)) {
            auto output_bytes = reinterpret_cast<const uint8_t*>(output);
            parseUPATaskProfiling(upa_taskList, output_bytes + offset.second, len, frc_speed_mhz, taskInfo);
        }
        if (offset.first == TaskInfo::ExecType::DPU && (type == TaskType::ALL || type == TaskType::DPU_SW)) {
            auto output_bytes = reinterpret_cast<const uint8_t*>(output);
            parseDPUTaskProfiling(dpu_taskList, output_bytes + offset.second, len, frc_speed_mhz, taskInfo);
        }
    }

    struct LayerTimes {
        LayerTimes() {
            dma_end_ns = 0;
            task_start_ns = std::numeric_limits<uint64_t>::max();
            task_wait_barriers_list = nullptr;
        }
        uint64_t dma_end_ns;
        uint64_t task_start_ns;
        const flatbuffers::Vector<uint32_t>* task_wait_barriers_list;
    };
    std::map<std::string, LayerTimes> layerInfoTimes;

    for (auto& task : taskInfo) {
        if (task.exec_type == TaskInfo::ExecType::DMA) {
            continue;
        }

        LayerTimes* layer;
        auto name = std::string(task.name);

        auto it = layerInfoTimes.find(name);
        if (it == layerInfoTimes.end()) {
            layerInfoTimes[name] = LayerTimes();
        }
        layer = &layerInfoTimes[name];

        if (task.start_time_ns < layer->task_start_ns) {
            layer->task_start_ns = task.start_time_ns;
            auto taskList = (task.exec_type == TaskInfo::ExecType::DPU) ? dpu_taskList : upa_taskList;
            layer->task_wait_barriers_list = (*taskList)[task.task_id]->associated_barriers()->wait_barriers();
        }
    }

    uint64_t min_dma_start_ns = std::numeric_limits<uint64_t>::max();
    uint64_t min_dpu_sw_start_ns = std::numeric_limits<uint64_t>::max();
    for (auto& task : taskInfo) {
        if (task.exec_type != TaskInfo::ExecType::DMA) {
            // Finding minimum start time for the DPU or SW task
            if (task.start_time_ns < min_dpu_sw_start_ns) {
                min_dpu_sw_start_ns = task.start_time_ns;
            }
            continue;
        }

        // Finding minimum start time for the DMA task
        if (task.start_time_ns < min_dma_start_ns) {
            min_dma_start_ns = task.start_time_ns;
        }

        // Finding DMA to DPU/SW timers synchronisation points.
        // DMA task should update and DPU task should wait for the same barrier within one layer
        auto task_end_ns = task.start_time_ns + task.duration_ns;
        LayerTimes* layer;
        auto name = std::string(task.name);

        auto it = layerInfoTimes.find(name);
        if (it == layerInfoTimes.end()) {
            continue;
        }
        layer = &layerInfoTimes[name];

        auto barriersList = (*dma_taskList)[task.task_id]->associated_barriers()->update_barriers();
        if (barriersList == nullptr || layer->task_wait_barriers_list == nullptr) {
            continue;
        }
        for (auto barrier : *layer->task_wait_barriers_list) {
            if (std::find((*barriersList).begin(), (*barriersList).end(), barrier) != (*barriersList).end()) {
                if (task_end_ns > layer->dma_end_ns) {
                    layer->dma_end_ns = task_end_ns;
                }
            }
        }
    }

    // Lets find the minimal offset between timers as we know for sure that DPU/SW task are started after the end of DMA
    // task because they connected via same barrier
    int64_t dma_task_timer_diff = 0;
    for (auto& times : layerInfoTimes) {
        if (times.second.dma_end_ns != 0 && times.second.task_start_ns != std::numeric_limits<uint64_t>::max()) {
            int64_t diff = (int64_t)times.second.dma_end_ns - times.second.task_start_ns;
            if (diff > dma_task_timer_diff) {
                dma_task_timer_diff = diff;
            }
        }
    }

    if (!dma_task_timer_diff) {
        // Could not calculate offset between timers(Most likely DMA profiling is disabled)
        // -> set offset based on begin time
        if (min_dma_start_ns != std::numeric_limits<uint64_t>::max()) {
            dma_task_timer_diff = min_dma_start_ns - min_dpu_sw_start_ns;
        } else {
            dma_task_timer_diff = -min_dpu_sw_start_ns;
        }
    }

    for (auto& task : taskInfo) {
        int64_t start_time_ns = task.start_time_ns;
        if (task.exec_type == TaskInfo::ExecType::DMA) {
            start_time_ns -= min_dma_start_ns;
        } else {
            start_time_ns += dma_task_timer_diff - min_dma_start_ns;
        }
        task.start_time_ns = (start_time_ns > 0) ? start_time_ns : 0;
    }
}

void vpux::profiling::getLayerInfo(const void* data, size_t data_len, const void* output, size_t output_len,
                                   std::vector<LayerInfo>& layerInfo) {
    std::vector<TaskInfo> taskInfo;

    getTaskInfo(data, data_len, output, output_len, taskInfo, TaskType::ALL);

    for (auto& task : taskInfo) {
        LayerInfo* layer;
        auto name = task.name;
        auto ptr = strstr(name, "/output tile");
        if (ptr != nullptr) {
            *ptr = '\0';
        }
        ptr = strstr(name, "/input");
        if (ptr != nullptr && strstr(name, "tile [") != nullptr) {
            *ptr = '\0';
        }

        auto result = std::find_if(begin(layerInfo), end(layerInfo), [&](LayerInfo item) {
            return strncmp(item.name, task.name, sizeof(LayerInfo::name)) == 0;
        });
        if (result == end(layerInfo)) {
            LayerInfo info = LayerInfo();
            strncpy(info.name, task.name, sizeof(LayerInfo::name));
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
}