#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("PermuteModel");
    mv::OpModel& om = unit.model();

    auto from_order = mv::Order("NCHW");
    auto to_order = mv::Order("NHWC");

    auto input0 = om.input({12,12,512,1}, mv::DType("Float16"), from_order, {{0},{1.0},{},{}}, "input0");
    auto permute0 = om.permute(input0, to_order, mv::DType("Float16"));
    om.output(permute0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
