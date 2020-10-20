#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("MaxpoolStridesExceedingSupported");
    mv::OpModel& om = unit.model();

    static const auto inf = std::numeric_limits<double>::infinity();

    //Input
    auto data_0 = om.input("input", {20,12,512,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4) /*NHWC*/);
    data_0->setQuantParams({{127},{0.007874016},{-1.000000000000000},{1.000000000000000},{0},{1}});

    const std::array<unsigned short, 2UL> maxpoolKSize = {20, 12};
    const std::array<unsigned short, 2UL> maxpoolStride = {20 ,12};
    auto maxpool = om.maxPool("maxpool", data_0, maxpoolKSize, maxpoolStride, {0, 0, 0, 0}, true);
    maxpool->setQuantParams({{127},{0.007874016},{-inf},{inf},{0},{1}});
    // Output
    auto output0 = om.output("", maxpool);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}