#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("InterpModel");
    mv::OpModel& om = unit.model();

    auto input0 = om.input({20,20,3,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{},{}}, "input0");
    //auto interp0 = om.interp(input0, 2.0, 0, 0 , 0, 0, false, mv::DType("Float16"));
    auto interp0 = om.interp(input0, 0.5, 0, 0 , 0, 0, true, mv::DType("Float16"));
    om.output(interp0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
