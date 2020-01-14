#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("ConvClampModel");
    mv::OpModel& om = unit.model();

    auto input0 = om.input({2,1,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4),  {{0},{1.0},{},{}}, "input#170");
    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (16, 1, 0);

    auto weights0 = om.constantInt(weightsData0,{1,1,16,1}, mv::DType("UInt8"), mv::Order::getRowMajorID(4), {{0},{1.0},{},{}});
    auto conv0 = om.conv(input0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1,  mv::DType("UInt8"),{{0},{1.0},{},{}} , "conv");
    om.output(conv0, mv::DType("Float16"), {{},{},{},{}});

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();
    compDesc.setPassArg("GlobalConfigParams", "DPUTasksinFloat", true);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

}
