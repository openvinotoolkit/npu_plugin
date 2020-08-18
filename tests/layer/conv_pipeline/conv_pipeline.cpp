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

    auto input0 = om.input({64,64,64,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4),  {{0},{0.00392156862745098},{},{}}, "input#170");

    auto pool0 = om.maxPool(input0, {1,1}, {1,1}, {0,0,0,0}, true, mv::DType("UInt8"), {{0},{0.00392156862745098},{},{}}, "pool");

    //NOTE: The idea here is that the first 64 elements of weights will be 1 and all the rest 0s. Doing that the whole output
    //tensor is expected to be full of 1s. Enable pipelining and validate...
    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (64, 255, 0);
    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (65472, 0, 0);
    weightsData0.insert(weightsData0.end(), weightsData1.begin(), weightsData1.end());

    auto weights0 = om.constantInt(weightsData0,{1,1,64,1024}, mv::DType("UInt8"), mv::Order::getRowMajorID(4), {{0},{0.00392156862745098},{},{}});
    auto conv0 = om.conv(pool0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1,  mv::DType("UInt8"),{{0},{0.00392156862745098},{},{}} , "conv");
    om.output(conv0);

    std::string compDescPath(argv[1]);
    printf("Using CD=%s\n", compDescPath.c_str());
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
