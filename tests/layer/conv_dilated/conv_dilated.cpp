#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{
    mv::CompilationUnit unit("ConvReluModel");
    mv::OpModel& om = unit.model();

    auto input0 = om.input({16,16,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4),  {{0},{1},{},{}}, "input#170");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (16*16*9/2, 1, 0);
    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (16*16*9/2, -1, 0);
    weightsData0.insert(weightsData0.end(), weightsData1.begin(), weightsData1.end());
    auto weights0 = om.constantInt(weightsData0,{3,3,16,16}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{0},{1.0},{},{}});
    //the 2 is dilation factor
    auto conv0 = om.conv(input0, weights0, {1, 1}, {0, 0, 0, 0}, 2, 1,  mv::DType("UInt8"),{{0},{1},{},{}} , "conv");
    //dilation 1
    //auto conv0 = om.conv(input0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1,  mv::DType("UInt8"),{{0},{1},{},{}} , "conv");
    om.output(conv0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.compilationDescriptor().remove("adapt", "PostTrainingQuantize");
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
