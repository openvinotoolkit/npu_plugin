#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{
    // TODO - Elu is HW activation funcition execute as PPE task, should be preceeded by conv (or other DPU task)

    mv::CompilationUnit unit("EluModel");
    mv::OpModel& om = unit.model();

    double alpha = 1;

    auto input0 = om.input({1,1,1000,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "input0");
    auto elu0 = om.elu(input0, alpha);
    om.output(elu0);
    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_MC-Prefetch1.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

}

