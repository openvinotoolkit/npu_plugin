#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("UnfusableScaleModel");
    mv::OpModel& om = unit.model();

    auto input0 = om.input({8,1,4,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4),  {{0},{1.0},{},{}}, "input#170");
    std::vector<int64_t> scalesData0 = mv::utils::generateSequence<int64_t> (4, 2, 2);

    auto scales0 = om.constantInt(scalesData0,{4}, mv::DType("UInt8"), mv::Order::getRowMajorID(1), {{0},{1.0},{},{}});
    auto scale = om.scale(input0, scales0, mv::DType("UInt8"), {{0},{1.0},{},{}}, "scale0");
    om.output(scale);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_MC-Prefetch1.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

}
