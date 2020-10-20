#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{
    mv::CompilationUnit unit("ReorgYoloDPUModel");
    mv::OpModel& om = unit.model();

    auto input0 = om.input("input#0", {16,16,64,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    input0->setQuantParams({{0},{1},{},{}});

    auto reorgYolo = om.reorgYolo("reorgYolo", input0, 4);
    reorgYolo->setQuantParams({{0},{1},{},{}});
    om.output("", reorgYolo);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    //unit.compilationDescriptor().remove("adapt", "PostTrainingQuantize");
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
