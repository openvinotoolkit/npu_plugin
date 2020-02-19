#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{
    mv::CompilationUnit unit("ConvFakeQuantizeModel");
    mv::OpModel& om = unit.model();

    std::vector<double> minData = mv::utils::generateSequence<double> (1, 0, 0);
    std::vector<double> maxData = mv::utils::generateSequence<double> (1, 255.0, 0);

    auto input0 = om.input({16,16,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4),  {{0},{1.0},{},{}}, "input#170");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (3 * 3 * 16, 1, 0);
    auto weights0 = om.constantInt(weightsData0,{3,3,16,1}, mv::DType("Float16"), mv::Order::getRowMajorID(4), {{0},{1.0},{},{}}, "Weights");

    auto weights_min0 = om.constant(minData,{1,1,1,1}, mv::DType("Float64"), mv::Order::getRowMajorID(4));
    auto weights_max0 = om.constant(maxData,{1,1,1,1}, mv::DType("Float64"), mv::Order::getRowMajorID(4));
    auto weights_min1 = om.constant(minData,{1,1,1,1}, mv::DType("Float64"), mv::Order::getRowMajorID(4));
    auto weights_max1 = om.constant(maxData,{1,1,1,1}, mv::DType("Float64"), mv::Order::getRowMajorID(4));

    auto weights_fake_quant = om.fakeQuantize(weights0, weights_min0, weights_max0, weights_min1, weights_max1, 256);

    auto conv0 = om.conv(input0, weights_fake_quant, {1, 1}, {0, 0, 0, 0}, 1, 1,  mv::DType("Default"), {{0},{1.0},{},{}}, "conv");

    om.output(conv0, mv::DType("Float16"));

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

}
