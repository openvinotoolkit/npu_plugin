#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("RegionYoloModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({13,13,125,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "input0");
    // Define Params
    unsigned coords = 4;
    unsigned classes = 20;
    bool do_softmax = true;
    unsigned num = 5;
    std::vector<unsigned> mask;
    auto regionyolo0 = om.regionYolo(input0, coords, classes, do_softmax, num, mask, mv::DType("Float16"));
    om.output(regionyolo0);
    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_MC-Prefetch1.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

}