#include "vpux/utils/core/profiling_parser.hpp"
#include "vpux/utils/core/error.hpp"

#include <map>

#include <flatbuffers/flatbuffers.h>
#include <schema/graphfile_generated.h>

using namespace vpux;

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

static const std::vector<std::pair<profiling_task_info::exec_type_t, uint32_t>> get_profilings_offets(
        const MVCNN::GraphFile* graphFile) {
    std::vector<std::pair<profiling_task_info::exec_type_t, uint32_t>> offsets;

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

            auto type = profiling_task_info::exec_type_t::NONE;
            if (name == "dma") {
                type = profiling_task_info::exec_type_t::DMA;
            } else if (name == "dpu") {
                type = profiling_task_info::exec_type_t::DPU;
            } else if (name == "upa") {
                type = profiling_task_info::exec_type_t::SW;
            }
            offsets.push_back({type, offset});
        }
    }

    return offsets;
}

static void parseDMATaskProfiling(const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* dma_taskList,
                                  const void* output, size_t output_len, double frc_speed_mhz,
                                  std::vector<profiling_task_info>& profInfo) {
    (void)output_len;
    auto output_bin = reinterpret_cast<const uint32_t*>(output);

    for (auto task : *dma_taskList) {
        if ((task->task_as_NNDMATask()->src()->name()->str() == "profilingInput:0") ||
            (task->task_as_NNDMATask()->src()->locale() == MVCNN::MemoryLocation_AbsoluteAddr)) {
            const auto task_name = task->name()->str();
            // std::cout << "Name:" << task_name << "\n";

            std::string profiling_meta[3];
            size_t pos = task_name.length();
            for (size_t i = 0; i < 3; i++) {
                size_t epos = pos;
                pos = task_name.rfind('_', epos);
                if (pos == std::string::npos) {
                    break;
                }
                profiling_meta[2 - i] = task_name.substr(pos + 1, epos - pos);
                pos--;
            }
            // std::cout << "Meta:" << profiling_meta[0] << "," << profiling_meta[1] << "," << profiling_meta[2] <<
            // "\n";
            if ((profiling_meta[2] != "PROFTASKBEGIN") && (profiling_meta[2] != "PROFBEGIN")) {
                unsigned layerNumber = 0;
                profiling_task_info profInfoItem;
                profInfoItem.layer_type[0] = '\0';
                profInfoItem.exec_type = profiling_task_info::exec_type_t::DMA;

                layerNumber = stoi(profiling_meta[2]);
                unsigned lastDMAid = stoi(profiling_meta[1]);
                auto currentDMAid = layerNumber * 2 - 1;

                // Use unsigned 32-bit arithmetic to automatically avoid overflow
                uint32_t diff = output_bin[currentDMAid] - output_bin[lastDMAid];
                auto taskName = task_name;
                taskName = taskName.substr(0, task_name.find("_PROF"));

                auto typeLen = sizeof(profInfoItem.name) / sizeof(profInfoItem.name[0]);
                auto length = taskName.copy(profInfoItem.name, typeLen, 0);
                profInfoItem.name[length] = '\0';
                // Convert to us //
                profInfoItem.start_time_ns = output_bin[lastDMAid] * 1000 / frc_speed_mhz;
                profInfoItem.duration_ns = (uint64_t)diff * 1000 / frc_speed_mhz;
                profInfoItem.task_id = layerNumber;

                profInfo.push_back(profInfoItem);
            }
        }
    }
}

static void parseUPATaskProfiling(const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* upa_taskList,
                                  const void* output, size_t output_len, double frc_speed_mhz,
                                  std::vector<profiling_task_info>& profInfo) {
    struct upa_data_t {
        uint64_t begin;
        uint64_t end;
        uint32_t stall_cycles;
        uint32_t active_cycles;
    };

    (void)output_len;
    auto output_upa = reinterpret_cast<const upa_data_t*>(output);

    unsigned currentPos = 0;
    for (auto task : *upa_taskList) {
        profiling_task_info profInfoItem;

        const auto taskName = task->name()->str();
        auto softLayer = task->task_as_UPALayerTask();
        if (softLayer != nullptr) {
            auto typeLen = sizeof(profInfoItem.layer_type) / sizeof(profInfoItem.layer_type[0]);
            strncpy(profInfoItem.layer_type, EnumNameSoftwareLayerParams(softLayer->softLayerParams_type()), typeLen);
        } else {
            profInfoItem.layer_type[0] = '\0';
        }
        profInfoItem.exec_type = profiling_task_info::exec_type_t::SW;
        uint64_t diff = output_upa[currentPos].end - output_upa[currentPos].begin;
        profInfoItem.start_time_ns = output_upa[currentPos].begin * 1000 / frc_speed_mhz;
        profInfoItem.duration_ns = diff * 1000 / frc_speed_mhz;
        profInfoItem.active_cycles = output_upa[currentPos].active_cycles;
        profInfoItem.stall_cycles = output_upa[currentPos].stall_cycles;
        profInfoItem.task_id = currentPos;

        auto typeLen = sizeof(profInfoItem.name) / sizeof(profInfoItem.name[0]);
        auto length = taskName.copy(profInfoItem.name, typeLen, 0);
        profInfoItem.name[length] = '\0';
        profInfo.push_back(profInfoItem);
        currentPos++;
    }
}

static void parseDPUTaskProfiling(const flatbuffers::Vector<flatbuffers::Offset<MVCNN::Task>>* dpu_taskList,
                                  const void* output, size_t output_len, double frc_speed_mhz,
                                  std::vector<profiling_task_info>& profInfo) {
    struct dpu_data_t {
        uint64_t begin;
        uint64_t end;
    };

    (void)output_len;
    auto output_upa = reinterpret_cast<const dpu_data_t*>(output);

    unsigned currentPos = 0;
    for (auto task : *dpu_taskList) {
        profiling_task_info profInfoItem;

        const auto taskName = task->name()->str();
        profInfoItem.layer_type[0] = '\0';
        profInfoItem.exec_type = profiling_task_info::exec_type_t::DPU;
        uint64_t diff = output_upa[currentPos].end - output_upa[currentPos].begin;
        profInfoItem.start_time_ns = output_upa[currentPos].begin * 1000 / frc_speed_mhz;
        profInfoItem.duration_ns = diff * 1000 / frc_speed_mhz;
        profInfoItem.active_cycles = 0;
        profInfoItem.stall_cycles = 0;
        profInfoItem.task_id = currentPos;

        auto typeLen = sizeof(profInfoItem.name) / sizeof(profInfoItem.name[0]);
        auto length = taskName.copy(profInfoItem.name, typeLen, 0);
        profInfoItem.name[length] = '\0';
        profInfo.push_back(profInfoItem);
        currentPos++;
    }
}

void vpux::getTaskProfilingInfo(const void* data, size_t data_len, const void* output, size_t output_len,
                                std::vector<profiling_task_info>& taskInfo, profiling_task_type type) {
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
    for (auto offset : offsets) {
        if (offset.first == profiling_task_info::exec_type_t::DMA &&
            (type == profiling_task_type::ALL || type == profiling_task_type::DMA)) {
            auto output_bytes = reinterpret_cast<const uint8_t*>(output);
            parseDMATaskProfiling(dma_taskList, output_bytes + offset.second, output_len - offset.second, frc_speed_mhz,
                                  taskInfo);
        }
        if (offset.first == profiling_task_info::exec_type_t::SW &&
            (type == profiling_task_type::ALL || type == profiling_task_type::DPU_SW)) {
            auto output_bytes = reinterpret_cast<const uint8_t*>(output);
            parseUPATaskProfiling(upa_taskList, output_bytes + offset.second, output_len - offset.second, frc_speed_mhz,
                                  taskInfo);
        }
        if (offset.first == profiling_task_info::exec_type_t::DPU &&
            (type == profiling_task_type::ALL || type == profiling_task_type::DPU_SW)) {
            auto output_bytes = reinterpret_cast<const uint8_t*>(output);
            parseDPUTaskProfiling(dpu_taskList, output_bytes + offset.second, output_len - offset.second, frc_speed_mhz,
                                  taskInfo);
        }
    }

    return;
}
void vpux::getLayerProfilingInfo(const void* data, size_t data_len, const void* output, size_t output_len,
                                 std::vector<profiling_layer_info>& layerInfo) {
    std::vector<profiling_task_info> taskInfo;

    getTaskProfilingInfo(data, data_len, output, output_len, taskInfo, vpux::profiling_task_type::ALL);

    for (auto& task : taskInfo) {
        profiling_layer_info* layer;
        auto name = task.name;
        auto ptr = strstr(name, "/output tile");
        if (ptr != nullptr) {
            *ptr = '\0';
        }

        auto result = std::find_if(begin(layerInfo), end(layerInfo), [&](profiling_layer_info item) {
            return strncmp(item.name, task.name, sizeof(profiling_layer_info::name)) == 0;
        });
        if (result == end(layerInfo)) {
            profiling_layer_info info = profiling_layer_info();
            strncpy(info.name, task.name, sizeof(profiling_layer_info::name));
            info.status = profiling_layer_info::layer_status_t::EXECUTED;
            info.start_time_ns = task.start_time_ns;
            layerInfo.push_back(info);
            layer = &layerInfo.back();
        } else {
            layer = result.base();
        }
        if (task.start_time_ns < layer->start_time_ns) {
            layer->start_time_ns = task.start_time_ns;
        }
        layer->duration_ns += task.duration_ns;

        if (task.exec_type == profiling_task_info::exec_type_t::DPU) {
            layer->dpu_ns += task.duration_ns;
            strncpy(layer->layer_type, task.layer_type, sizeof(profiling_layer_info::layer_type));
        }
        if (task.exec_type == profiling_task_info::exec_type_t::SW) {
            layer->sw_ns += task.duration_ns;
            strncpy(layer->layer_type, task.layer_type, sizeof(profiling_layer_info::layer_type));
        }
        if (task.exec_type == profiling_task_info::exec_type_t::DMA) {
            layer->dma_ns += task.duration_ns;
        }
    }
}