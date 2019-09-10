//This file is the parsed network which is created through python.
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include "iostream"
#include "fstream"

int main()
{
    std::string path = std::getenv("MDK_HOME");
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({12,12,32,1}, mv::DType("Float64"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "input:0#3");

    std::vector<double> weightsData0 = mv::utils::generateSequence<double> (1*1*32*64);
    auto weights0 = om.constant(weightsData0,{1,1,32,64}, mv::DType("Float64"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "output/Conv2D#1_weights#2");
    auto conv0 = om.conv(input0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float64"), {{0},{1.0},{-inf},{inf}}, "output/Conv2D:0#4");

    om.output(conv0);

    std::string compDescPath = "/home/tbartsok/Desktop/WORK/mcmCompiler/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
