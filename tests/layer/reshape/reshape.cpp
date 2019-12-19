//This file is the parsed network which is created through python.
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>

int main()
{
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({7,1,8960,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "input:0#9");
//    auto input0 = om.input({7,7,1280,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "input:0#9");

//    auto reshape0 = om.reshape(input0, mv::Shape({7, 1, 8960, 1}), mv::DType("UInt8"), {{0},{1.0},{-inf},{inf}}, "Reshape:0#10");

    auto pool0 = om.averagePool(input0, {7, 1}, {1, 1}, {0, 0, 0, 0}, true, "", "floor", mv::DType("UInt8"), {{0},{1.0},{-inf},{inf}}, "average1:0#11");

//    auto reshape1 = om.reshape(pool0, mv::Shape({7, 1, 1280, 1}), mv::DType("UInt8"), {{0},{1.0},{-inf},{inf}}, "Reshape_1:0#12");

//    auto reshape2 = om.reshape(reshape1, mv::Shape({7, 1, 1280, 1}), mv::DType("UInt8"), {{0},{1.0},{-inf},{inf}}, "Reshape_2:0#13");

//    auto pool1 = om.averagePool(reshape2, {7, 1}, {1, 1}, {0, 0, 0, 0}, true, "", "floor", mv::DType("UInt8"), {{0},{1.0},{-inf},{inf}}, "average2:0#14");

//    auto reshape3 = om.reshape(pool1, mv::Shape({1, 1, 1280, 1}), mv::DType("UInt8"), {{0},{1.0},{-inf},{inf}}, "Reshape_3:0#15");

//    std::vector<int64_t> fcweightsData0 = mv::utils::generateSequence<int64_t> (1280*1000);
//    auto weights0 = om.constantInt(fcweightsData0,{1280,1000}, mv::DType("UInt8"), mv::Order("WC"), {{0},{1.0},{-inf},{inf}}, "output_weights#8");
//    auto fc0 = om.fullyConnected(reshape3, weights0, mv::DType("UInt8"), {{0},{1.0},{-inf},{inf}}, "output:0#16");

    om.output(pool0);

    std::string compDescPath = "/home/tbartsok/Desktop/WORK/kmb-plugin/thirdparty/movidius/mcmCompiler/config/compilation/release_kmb_MC-Prefetch1.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
