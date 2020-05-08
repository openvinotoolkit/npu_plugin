#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("EltwiseReplacement");
    mv::OpModel& om = unit.model();

    static const auto inf = std::numeric_limits<double>::infinity();

    //Input
    auto data_0 = om.input({24,24,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4) /*NHWC*/,  {{127},{0.007874016},{-1.000000000000000},{1.000000000000000},{0},{1}}, "input");

    const std::array<unsigned short, 2UL> maxpoolKSize = {3,3};
    const std::array<unsigned short, 2UL> maxpoolStride = {10 ,10};
    auto maxpool = om.maxPool(data_0,maxpoolKSize,maxpoolStride,{0, 0, 0, 0}, true,mv::DType("UInt8"),{{127},{0.007874016},{-inf},{inf},{0},{1}} , "maxpool");
    // Output
    auto output0 = om.output(maxpool);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}