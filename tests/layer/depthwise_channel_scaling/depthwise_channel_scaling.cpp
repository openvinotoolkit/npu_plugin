#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{
    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({4, 4, 3, 1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{0},{1.0},{},{}});
    std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(3, 1, 0);
    // per channel scale
    const std::vector<int64_t> zp(3, 0);
    std::vector<double> scale(3);
    for (std::size_t i = 0; i < 3; i++)
    {
        if (i%2==0)
        {
            scale[i] = 1.0;
        }
        else
        {
            scale[i] = 2.0;
        }
    }

    auto weights1 = om.constantInt(weightsData, {1, 1, 3, 1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{0},{scale},{},{}});
    auto depthwiseConv = om.depthwiseConv(input, weights1, {1, 1}, {0, 0, 0, 0}, 1, mv::DType("UInt8"), {{0},{1.0},{},{}});
    auto output = om.output(depthwiseConv);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_MC-Prefetch1.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}