//This file is the parsed network which is created through python.
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"
#include "iostream"
#include "fstream"
int main()
{
    std::string path = std::getenv("MCM_HOME");
    double inf = std::numeric_limits<double>::infinity();
    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();

    auto input0 = om.input({1,1,40257,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "input0");
    mv::Shape new_shape({21,1,1917,1});
    auto reshape0 = om.reshape(input0, new_shape, input0->getOrder().toString(), mv::DType("Float16"));
    om.output(reshape0);
    
    std::string compDescPath = path + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
