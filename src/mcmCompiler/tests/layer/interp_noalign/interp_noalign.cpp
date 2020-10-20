#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("InterpModel");
    mv::OpModel& om = unit.model();

    auto input0 = om.input("input0", {20,20,3,1}, mv::DType("Float16"), mv::Order::getZMajorID(4));
    input0->setQuantParams({{0},{1.0},{},{}});
    auto interp0 = om.interp("", input0, 2.0, 0, 0 , 0, 0, false);
    interp0->setQuantParams({{0},{1.0},{},{}});
    om.output("", interp0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
