#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("InterpModel");
    mv::OpModel& om = unit.model();

    auto input0 = om.input("input0", {45,60,512,1}, mv::DType("Float16"), mv::Order::getZMajorID(4));
    //auto interp0 = om.interp("", input0, 1.0, 0, 0 , 0, 0, false);
    auto interp0 = om.interp("", input0, 1.0, 0, 0 ,30,23, true);
    om.output("", interp0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
