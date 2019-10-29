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

    double min = -0.5;
    double max = 0.5;

    mv::Data::TensorIterator input0 = om.input({1,1,32,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "input0");
    auto clamp0 = om.clamp(input0, min, max);
    om.output(clamp0);
    
    std::string compDescPath = path + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
