#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{
    mv::CompilationUnit unit("FullyConnectedChannelScalingModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({1, 1, 16, 1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{0},{1.0},{},{}});

    // per channel scale
    const std::vector<int64_t> zp(100, 0);
    std::vector<double> scale(100);
    for (std::size_t i = 0; i < 100; i++)
    {
        if (i%2==0)
        {
            scale[i] = 1;
        }
        else
        {
            scale[i] = 0.5;
        }
    }

    std::vector<double> weightsData = mv::utils::generateSequence<double>(input->getShape()[2] * 100u, 1, 0);
    auto weights1 = om.constant(weightsData, {input->getShape()[2], 100}, mv::DType("Float64"), mv::Order::getColMajorID(2), {{0},{scale},{},{}});
    auto fullyConnected = om.fullyConnected(input, weights1, mv::DType("UInt8"), {{0},{1.0},{},{}});
    auto output = om.output(fullyConnected);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.compilationDescriptor().remove("adapt", "PostTrainingQuantize");
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
