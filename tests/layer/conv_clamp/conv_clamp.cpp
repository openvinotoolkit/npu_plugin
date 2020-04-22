#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("ConvClampModel");
    mv::OpModel& om = unit.model();

    // ASSUMPTION: we want the output of the convolution to be in the REAL range
    // [20, 40]. This means that we have to use a maximum operation for the
    // lower bound and and a minimum operation with the upper bound

    // ASSUMPTION2: Minimum goes always first otherwise fusing will be wrong
    double lowerBound = 20;
    double upperBound = 40;

    auto input0 = om.input({2,1,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4),  {{0},{1.0},{},{}}, "input#170");
    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (16, 1, 0);

    auto weights0 = om.constantInt(weightsData0,{1,1,16,1}, mv::DType("UInt8"), mv::Order::getRowMajorID(4), {{0},{1.0},{},{}});
    auto conv0 = om.conv(input0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1,  mv::DType("UInt8"),{{0},{1.0},{},{}} , "conv");
    auto min0 = om.minimum(conv0, upperBound);
    auto max0 = om.maximum(min0, lowerBound);
    om.output(max0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_MC-PrefetchAdaptive.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.compilationDescriptor().remove("adapt", "PostTrainingQuantize");
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

}
