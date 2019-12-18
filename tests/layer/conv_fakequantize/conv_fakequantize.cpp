#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{
    mv::CompilationUnit unit("ConvFakeQuantizeModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({2,1,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4),  {{0},{1.0},{},{}}, "input#170");
    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (16, 1, 0);
    auto weights0 = om.constantInt(weightsData0,{1,1,16,1}, mv::DType("UInt8"), mv::Order::getRowMajorID(4), {{0},{1.0},{},{}});
    auto conv0 = om.conv(input0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1,  mv::DType("Float16"),{{0},{1.0},{},{}} , "conv");

    std::vector<double> minData = mv::utils::generateSequence<double> (1, 0, 0);
    auto min0 = om.constant(minData,{1,1,1,1}, mv::DType("Float64"), mv::Order::getRowMajorID(4));
    std::vector<double> maxData = mv::utils::generateSequence<double> (1, 255.0, 0);
    auto max0 = om.constant(maxData,{1,1,1,1}, mv::DType("Float64"), mv::Order::getRowMajorID(4));
    auto min1 = om.constant(minData,{1,1,1,1}, mv::DType("Float64"), mv::Order::getRowMajorID(4));
    auto max1 = om.constant(maxData,{1,1,1,1}, mv::DType("Float64"), mv::Order::getRowMajorID(4));

    auto fake_quant0 = om.fakeQuantize(conv0, min0, max0, min1, max1, 256);
    om.output(fake_quant0);
    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_MC-Prefetch1.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

}
