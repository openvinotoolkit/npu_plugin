#include <iostream>
#include <fstream>
#include <sys/stat.h>

#include "mcm/utils/profiling_parser.hpp"

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

    std::vector<mv::utils::prof_info_t> deviceProfiling;
    mv::utils::prof_total_info_t prof_total_info;
    mv::utils::getProfilingInfo(blob_bin.data(), output_bin.data(), deviceProfiling, &prof_total_info);

    for (auto& task : deviceProfiling ) {
        std::cout << "Task(" << task.start_layer_id <<"->" << task.end_layer_id << "): " << task.name << " Time: " << task.time << std::endl;
    }
    std::cout << "TotalTime: " << prof_total_info.time << std::endl;

}