#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("ReorgYoloModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({125,13,13,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "input0");

    unsigned stride = 1;

    auto reorgyolo0 = om.reorgYolo(input0, stride, mv::DType("Float16"));
    om.output(reorgyolo0);

    //input shape: {125, 13, 13, 1}
    //output shape: {21125, 1}

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_MC-Prefetch1.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
    
}
