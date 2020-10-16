#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"

#include <iostream>
#include <fstream>
#include <sys/stat.h>

void print_buffer_info(mv::BufferEntry parsedBuffer)
{
    std::cout << "  name: " << parsedBuffer.getName() << std::endl;
    std::cout << "  type: " << (int)parsedBuffer.getBufferType() << std::endl;
    std::cout << "  size: " << parsedBuffer.getSize() << std::endl;
    std::cout << "  shape: " << parsedBuffer.getShape().toString() << std::endl;
    std::cout << "  order: " << parsedBuffer.getOrder().toString() << std::endl;
    std::cout << "  dtype: " << parsedBuffer.getDType().toString() << std::endl;
    std::cout << std::endl;
}

int main(int argc, char** argv)
{

    // Compilation unit must be loaded first
    mv::CompilationUnit unused("unused");

    if (argc < 2)
    {
        std::cout << "Usage: parse_blob <filename>" << std::endl << std::endl;
        return 0;
    }

    fstream blob_file;
    std::string blob_filename(argv[1]);

    struct stat results;
    auto blob_size = 0;
    if (stat(blob_filename.c_str(), &results) != 0)
    {
        std::cout << "Error reading blob!" << std::endl;
        return -1;
    }
    else
    {
        blob_size = results.st_size;
    }

    std::vector<char> blob_bin(results.st_size);
    blob_file.open(blob_filename, ios::in | ios::binary);
    blob_file.read(&blob_bin[0], results.st_size);

    std::string targetDescPath = mv::utils::projectRootPath() + "/config/target/release_kmb.json";
    auto targetDesc = mv::TargetDescriptor(targetDescPath);
    mv::CompilationUnit unit(blob_bin.data(), blob_size, targetDesc);

    char name[256];
    unit.getName(name, 256);
    auto inputs = unit.getBufferMap().getInputCount();
    auto outputs = unit.getBufferMap().getOutputCount();
    auto scratches = unit.getBufferMap().getScratchCount();
    auto profilers = unit.getBufferMap().getProfilingCount();

    std::cout << std::endl << "Model: " << name << std::endl << std::endl;

    std::cout << "Input buffers:" << std::endl;
    for (auto i=0; i<inputs; ++i)
        print_buffer_info(unit.getBufferMap().getInput()[i]);

    std::cout << "Output buffers:" << std::endl;
    for (auto i=0; i<outputs; ++i)
        print_buffer_info(unit.getBufferMap().getOutput()[i]);

    std::cout << "Scratch buffers:" << std::endl;
    for (auto i = 0; i < scratches; ++i)
        print_buffer_info(unit.getBufferMap().getScratch()[i]);

    std::cout << "Profiling buffers:" << std::endl;
    for (auto i = 0; i < profilers; ++i)
        print_buffer_info(unit.getBufferMap().getProfiling()[i]);

    return 0;
}
