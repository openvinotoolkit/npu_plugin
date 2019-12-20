#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("SoftmaxModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({1,1,1000,1}, mv::DType("Float64"), mv::Order::getZMajorID(4), {{0},{1.0},{},{}}, "input:0#4");
    //auto input0 = om.input({1,21,1917,1}, mv::DType("Float64"), mv::Order::getZMajorID(4), {{0},{1.0},{},{}}, "input:0#4");
    std::string axis = "C";
    //std::string axis = "H";
    auto softmax0 = om.softmax(input0, axis);
    om.output(softmax0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_MC-Prefetch1.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

}
