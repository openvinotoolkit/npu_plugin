#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("ConvClampModel");
    mv::OpModel& om = unit.model();

    auto input0 = om.input("input#170", {2,1,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4));
    input0->setQuantParams({{0},{1},{},{}});

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (16, 1, 0);
    auto weights0 = om.constantInt("", weightsData0, {1,1,16,1}, mv::DType("UInt8"), mv::Order::getRowMajorID(4));
    auto conv0 = om.conv("conv", input0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1);
    weights0->setQuantParams({{0},{1},{},{}});
    conv0->setQuantParams({{0},{1},{},{}});
    om.output("", conv0, mv::DType("Float16"));

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();
    unit.compilationDescriptor().remove("adapt", "PostTrainingQuantize");

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
