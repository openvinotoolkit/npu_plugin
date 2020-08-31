#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main(int argc, char **argv)
{

    if (argc != 2) {
      fprintf(stderr, "./conv_pipeline <comp-descriptor-file>\n");
      exit(1);
    }


    mv::CompilationUnit unit("ConvPipeline");
    mv::OpModel& om = unit.model();

    auto input0 = om.input({7,7,2048,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4),  {{0},{0.00392156862745098},{},{}}, "input#170");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (512*2048, 0, 0);
    auto weights0 = om.constantInt(weightsData0,{1,1,2048,512}, mv::DType("UInt8"), mv::Order::getRowMajorID(4), {{0},{0.00392156862745098},{},{}});
    auto conv0 = om.conv(input0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1,  mv::DType("UInt8"),{{0},{0.00392156862745098},{},{}} , "conv0");
    
    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (512*512*3*3, 0, 0);
    auto weights1 = om.constantInt(weightsData1,{3,3,512,512}, mv::DType("UInt8"), mv::Order::getRowMajorID(4), {{0},{0.00392156862745098},{},{}});
    auto conv1 = om.conv(conv0, weights1, {1, 1}, {0, 0, 0, 0}, 1, 1,  mv::DType("UInt8"),{{0},{0.00392156862745098},{},{}} , "conv1");
    
    std::vector<int64_t> weightsData2 = mv::utils::generateSequence<int64_t> (512*2048, 255, 0);
    auto weights2 = om.constantInt(weightsData2,{1,1,512,2048}, mv::DType("UInt8"), mv::Order::getRowMajorID(4), {{0},{0.00392156862745098},{},{}});
    auto conv2 = om.conv(conv1, weights2, {1, 1}, {0, 0, 0, 0}, 1, 1,  mv::DType("UInt8"),{{0},{0.00392156862745098},{},{}} , "conv2");
    
    om.output(conv2);

    std::string compDescPath(argv[1]);
    printf("Using CD=%s\n", compDescPath.c_str());
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
