//This file is the parsed network which is created through python.
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
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
    auto input0 = om.input({28,28,512,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#3");

    auto pool0 = om.maxPool(input0, {1, 1}, {1, 1}, {0, 0, 0, 0}, true, "", "floor", {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "pool/max_pool#4");

    auto pool1 = om.maxPool(input0, {1, 1}, {1, 1}, {0, 0, 0, 0}, true, "", "floor", {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "pool_1/max_pool#5");

    auto eltwise0 = om.add(pool0,pool1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "eltwise_1#6");

    om.output(eltwise0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
