#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("ReshapeModel");
    mv::OpModel& om = unit.model();

    auto input0 = om.input({1,1,40257,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{},{}}, "input0");
    mv::Shape new_shape({21,1,1917,1});
    auto reshape0 = om.reshape(input0, new_shape, input0->getOrder().toString(), mv::DType("Float16"));
    om.output(reshape0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_MC-Prefetch1.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

}
