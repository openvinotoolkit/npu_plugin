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

void mv::utils::getProfilingInfo(const void* data, const void* output, std::vector<prof_info_t>& profInfo,
                      prof_total_info_t* prof_total_info) {
    if ((nullptr == data) || (nullptr == output)) {
        throw mv::ArgumentError("profiling", "profiling", "0", "Empty input data");
    }

    auto output_bin = reinterpret_cast<const uint32_t*>(output);

    const auto* graphFilePtr = MVCNN::GetGraphFile(data);
    MVCNN::GraphFileT graphFile;
    graphFilePtr->UnPackTo(&graphFile);

    /* Finding of DMA task list */
    std::vector<std::unique_ptr<MVCNN::TaskT>>* dma_taskList = nullptr;
    std::vector<std::unique_ptr<MVCNN::TaskT>>* dpu_upa_taskList = nullptr;
    for (auto& task_list_item : graphFile.task_lists) {
        if (task_list_item->content[0]->task.type == MVCNN::SpecificTask_NNDMATask) {
            dma_taskList = &task_list_item->content;
        }
        if (task_list_item->content[0]->task.type == MVCNN::SpecificTask_NCE2Task ||
            task_list_item->content[0]->task.type == MVCNN::SpecificTask_UPALayerTask) {
            dpu_upa_taskList = &task_list_item->content;
        }
        if ((dma_taskList != nullptr) && (dpu_upa_taskList != nullptr))
            break;
    }
    if ((nullptr == dma_taskList) || (nullptr == dpu_upa_taskList)) {
        mv::MasterError("profiling", "Cound not find task list");
    }

    std::map<std::string, execution_info_t> dpu_upa_types;

    for (auto& task : *dpu_upa_taskList) {
        if (task->task.type == MVCNN::SpecificTask_NCE2Task) {
            auto item = task->task.AsNCE2Task();
            dpu_upa_types[task->name] = {EnumNameDPULayerType(item->invariant->dpu_task_type), "DPU"};
        } else if (task->task.type == MVCNN::SpecificTask_UPALayerTask) {
            auto item = task->task.AsUPALayerTask();
            dpu_upa_types[task->name] = {EnumNameSoftwareLayerParams(item->softLayerParams.type), "UPA"};
        }
    }

    std::vector<unsigned> layerNumbers;

    uint64_t lastTime = 0;
    uint64_t beginTime = 0;
    unsigned currentPos = 0;
    for (auto& task : *dma_taskList) {
        if (task->task.AsNNDMATask()->src->name == "profilingInput:0") {
            auto str = task->name;
            auto pos = str.rfind("_");
            unsigned layerNumber = stoi(str.substr(pos + 1));
            layerNumbers.push_back(layerNumber);

            str = str.substr(0, pos);
            pos = str.rfind("_");
            unsigned lastDMAid = stoi(str.substr(pos + 1));

            if (task->name.find("_PROFBEGIN") != std::string::npos) {
                lastTime = output_bin[currentPos];
                beginTime = lastTime;
            } else {
                /* Use unsigned 32-bit arithmetic to automatically avoid overflow */
                uint32_t diff = output_bin[currentPos] - output_bin[lastDMAid];
                auto taskName = task->name;
                taskName = taskName.substr(0, task->name.find("_PROF"));
                lastTime = output_bin[currentPos];
                prof_info_t profInfoItem;
                profInfoItem.name = taskName;
                /*Convert to us (FRC is 500MHz) */
                profInfoItem.time = diff / 500;
                profInfoItem.start_layer_id = layerNumbers[lastDMAid];
                profInfoItem.end_layer_id = layerNumber;

                auto layer_type = dpu_upa_types.find(taskName);
                if (layer_type != dpu_upa_types.end()) {
                    profInfoItem.layer_type = layer_type->second.layer_type;
                    profInfoItem.exec_type = layer_type->second.hw_type;
                }

                profInfo.push_back(profInfoItem);
            }

            currentPos++;
        }
    }
    if (lastTime < beginTime)
        lastTime += 0x100000000;
    if (prof_total_info)
        prof_total_info->time = lastTime - beginTime;
}