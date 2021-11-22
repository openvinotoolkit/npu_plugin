#include <iostream>
#include <iomanip>
#include <fstream>
#include <sys/stat.h>

#include "vpux/utils/plugin/profiling_parser.hpp"

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cout << "Usage: prof_parser <blob path> <output.bin path>" << std::endl;
        return 0;
    }
    std::string blobPath(argv[1]);
    std::string outputPath(argv[2]);

    struct stat blob_stat, output_stat;
    if (stat(blobPath.c_str(), &blob_stat) != 0) {
        std::cout << "Error reading blob!" << std::endl;
        return -1;
    }

    if (stat(outputPath.c_str(), &output_stat) != 0) {
        std::cout << "Error reading output!" << std::endl;
        return -1;
    }
    
    std::ifstream blob_file;
    std::vector<unsigned char> blob_bin(blob_stat.st_size);
    blob_file.open(blobPath, std::ios::in | std::ios::binary);
    blob_file.read((char*)blob_bin.data(), blob_stat.st_size);
    blob_file.close();

    std::fstream output_file;
    std::vector<uint32_t> output_bin(output_stat.st_size/4);
    output_file.open(outputPath, std::ios::in | std::ios::binary);
    output_file.read((char*)output_bin.data(), output_stat.st_size);
    output_file.close();

    std::vector<vpux::ProfilingTaskInfo> taskProfiling;
    vpux::getTaskProfilingInfo(blob_bin.data(), blob_bin.size(), output_bin.data(), output_bin.size(), 
        taskProfiling, vpux::ProfilingTaskType::ALL);

    uint64_t last_time_ns = 0;
    for (auto& task : taskProfiling) {
        std::string exec_type_str;
        switch (task.exec_type) {
            case vpux::ProfilingTaskInfo::exec_type_t::DMA:
                exec_type_str = "DMA";
                std::cout << "Task(" << exec_type_str << "): " << std::setw(50) << task.name << "\tTime: " << (float)task.duration_ns/1000 << "\tStart: " << task.start_time_ns/1000 << std::endl;
                break;
            case vpux::ProfilingTaskInfo::exec_type_t::DPU:
                exec_type_str = "DPU";
                std::cout << "Task(" << exec_type_str << "): " << std::setw(50) << task.name << "\tTime: " << (float)task.duration_ns/1000 << "\tStart: " << task.start_time_ns/1000 << std::endl;
                break;
            case vpux::ProfilingTaskInfo::exec_type_t::SW:
                exec_type_str = "SW";
                std::cout << "Task(" << exec_type_str << "): " << std::setw(50) << task.name << "\tTime: " << (float)task.duration_ns/1000
                << "\tCycles:" << task.active_cycles << "(" << task.stall_cycles << ")" << std::endl;
                break;
        }

        uint64_t task_end_time_ns = task.start_time_ns + task.duration_ns;
        if (last_time_ns < task_end_time_ns) {
            last_time_ns = task_end_time_ns;
        }
    }

    std::vector<vpux::ProfilingLayerInfo> layerProfiling;
    vpux::getLayerProfilingInfo(blob_bin.data(), blob_bin.size(), output_bin.data(), output_bin.size(), 
        layerProfiling);
    uint64_t total_time = 0;
    for (auto& layer : layerProfiling ) {
        std::cout << "Layer: " << std::setw(30) << layer.name 
            << " DPU: " << std::setw(5) << layer.dpu_ns/1000
            << " SW: " << std::setw(5) << layer.sw_ns/1000
            << " DMA: " << std::setw(5) << layer.dma_ns/1000
            << "\tStart: " << layer.start_time_ns/1000 << std::endl;
        total_time += layer.dpu_ns + layer.sw_ns + layer.dma_ns;
    }

    std::cout << "TotalTime: " << total_time/1000 << "us, Real: " << last_time_ns/1000 << "us" << std::endl;
    return 0;
}
