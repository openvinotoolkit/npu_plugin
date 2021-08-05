#include "mcm/utils/profiling_parser.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/base/exception/master_error.hpp"

#include <map>

#include <flatbuffers/flatbuffers.h>
#include <schema/graphfile/graphfile_generated.h>

typedef struct {
    std::string layer_type;
    std::string hw_type;
} execution_info_t;

void mv::utils::getProfilingInfo(const void* data, const void* output, std::vector<ProfInfo>& profInfo,
                      ProfTotalInfo* prof_total_info) {
    if ((nullptr == data) || (nullptr == output)) {
        throw mv::ArgumentError("profiling", "profiling", "0", "Empty input data");
    }

    auto output_bin = reinterpret_cast<const uint32_t*>(output);

    const auto* graphFilePtr = MVCNN::GetGraphFile(data);
    MVCNN::GraphFileT graphFile;
    graphFilePtr->UnPackTo(&graphFile);

    double frc_speed_mhz = 0; 
    for (auto& frequency : graphFile.header->resources->processor_frequencies) {
        if (frequency->item == MVCNN::PhysicalProcessor_NCE_Cluster) {
            frc_speed_mhz = frequency->number;
            break;
        }
    }

    if (!frc_speed_mhz) {
        switch (graphFile.header->device) {
            case MVCNN::TargetDevice::TargetDevice_KMB:
                switch (graphFile.header->device_revision) {
                    case MVCNN::TargetDeviceRevision::TargetDeviceRevision_A0:
                        frc_speed_mhz = 500;
                        break;
                    case MVCNN::TargetDeviceRevision::TargetDeviceRevision_B0:
                        frc_speed_mhz = 700;
                        break;
                    default:
                        throw mv::ArgumentError("profiling", "TargetDeviceRevision", 
                            EnumNameTargetDeviceRevision(graphFile.header->device_revision), "value is not supported");
                }
                break;
            case MVCNN::TargetDevice::TargetDevice_TBH:
                frc_speed_mhz = 700;
                break;
            default:
                throw mv::ArgumentError("profiling", "TargetDevice", 
                            EnumNameTargetDevice(graphFile.header->device), "value is not supported");
        }
    }
    if (!frc_speed_mhz) frc_speed_mhz = 500;

    // Finding of DMA task list //
    std::vector<std::unique_ptr<MVCNN::TaskT>>* dma_taskList = nullptr;
    std::vector<std::vector<std::unique_ptr<MVCNN::TaskT>>*> dpu_upa_taskList;
    for (auto& task_list_item : graphFile.task_lists) {
        if (task_list_item->content[0]->task.type == MVCNN::SpecificTask_NNDMATask) {
            dma_taskList = &task_list_item->content;
        }
        if (task_list_item->content[0]->task.type == MVCNN::SpecificTask_NCE2Task ||
            task_list_item->content[0]->task.type == MVCNN::SpecificTask_UPALayerTask) {
            dpu_upa_taskList.push_back(&task_list_item->content);
        }
    }
    if ((nullptr == dma_taskList) || (!dpu_upa_taskList.size())) {
        mv::MasterError("profiling", "Cound not find task list");
    }

    std::map<std::string, execution_info_t> dpu_upa_types;

    for (auto taskList : dpu_upa_taskList) {
        for (auto& task : *taskList) {
            if (task->task.type == MVCNN::SpecificTask_NCE2Task) {
                auto item = task->task.AsNCE2Task();
                dpu_upa_types[task->name] = {EnumNameDPULayerType(item->invariant->dpu_task_type), "DPU"};
            } else if (task->task.type == MVCNN::SpecificTask_UPALayerTask) {
                auto item = task->task.AsUPALayerTask();
                dpu_upa_types[task->name] = {EnumNameSoftwareLayerParams(item->softLayerParams.type), "UPA"};
            }
        }
    }

    std::vector<unsigned> layerNumbers;

    std::map<std::string, int> layerNames;

    uint64_t lastTime = 0;
    uint64_t beginTime = 0;
    unsigned currentPos = 0;
    for (auto& task : *dma_taskList) {
        if ((task->task.AsNNDMATask()->src->name == "profilingInput:0") 
        || (task->task.AsNNDMATask()->src->locale == MVCNN::MemoryLocation_AbsoluteAddr)) {
            auto str = task->name;
            auto pos = str.rfind('_');
            unsigned layerNumber = stoi(str.substr(pos + 1));
            layerNumbers.push_back(layerNumber);

            str = str.substr(0, pos);
            pos = str.rfind('_');
            unsigned lastDMAid = stoi(str.substr(pos + 1));

            if (task->name.find("_PROFBEGIN") != std::string::npos) {
                lastTime = output_bin[currentPos];
                beginTime = lastTime;
            } else {
                ProfInfo profInfoItem;
                /* Use unsigned 32-bit arithmetic to automatically avoid overflow */
                uint32_t diff = output_bin[currentPos] - output_bin[lastDMAid];
                auto taskName = task->name;
                taskName = taskName.substr(0, task->name.find("_PROF"));

                if (taskName[0] == '[') {
                    auto cpos = task->name.find('_');
                    auto epos = task->name.find(']');
                    if (epos != std::string::npos) {
                        profInfoItem.layer_type = taskName.substr(1, cpos-1);
                        auto dotpos = profInfoItem.layer_type.find('.');
                        if (dotpos != std::string::npos) {
                            profInfoItem.layer_type = profInfoItem.layer_type.substr(dotpos+1);
                        }
                        profInfoItem.exec_type = taskName.substr(cpos+1, epos-cpos-1);

                        taskName = "["+profInfoItem.layer_type+"]"+taskName.substr(epos+1);
                    }
                } else {
                    auto layer_type = dpu_upa_types.find(taskName);
                    if (layer_type != dpu_upa_types.end()) {
                        profInfoItem.layer_type = layer_type->second.layer_type;
                        profInfoItem.exec_type = layer_type->second.hw_type;
                    }
                }

                // Prevent existence of the same layer names 
                auto layerNameIt = layerNames.find(taskName);
                if (layerNameIt != layerNames.end()) {
                    layerNames[taskName]++;
                    taskName += layerNames[taskName];
                } else layerNames[taskName] = 0;

                lastTime = output_bin[currentPos];
                profInfoItem.name = taskName;
                // Convert to us //
                profInfoItem.time = diff / frc_speed_mhz;
                profInfoItem.start_layer_id = layerNumbers[lastDMAid];
                profInfoItem.end_layer_id = layerNumber;

                profInfo.push_back(profInfoItem);
            }

            currentPos++;
        }
    }
    if (lastTime < beginTime)
        lastTime += 0x100000000;
    if (prof_total_info)
        prof_total_info->time = (lastTime - beginTime) / frc_speed_mhz;
}