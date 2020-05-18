#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("MaxpoolStridesExceedingSupported");
    mv::OpModel& om = unit.model();

    static const auto inf = std::numeric_limits<double>::infinity();

    //Input
    auto data_0 = om.input({30,30,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4) /*NHWC*/,  {{127},{0.007874016},{-1.000000000000000},{1.000000000000000},{0},{1}}, "input");
    const std::array<unsigned short, 2UL> stride = {1 ,10};

    const uint8_t zeroPointWt =8;
    const mv::Shape kernel = mv::Shape({3,3,16,16});
    std::vector<int64_t> weightsData0(kernel.totalSize(), zeroPointWt+zeroPointWt);

    auto d_weights0 = om.constantInt(weightsData0,kernel, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{145},{0.006896552},{-0.2301538735628128},{0.17448118329048157}}, "dwconv0#0_weights#1");
    auto depthConv0 = om.depthwiseConv(data_0, d_weights0, stride, {0, 0, 0, 0}, 1, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "dwconv0#4");

    auto output0 = om.output(depthConv0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}