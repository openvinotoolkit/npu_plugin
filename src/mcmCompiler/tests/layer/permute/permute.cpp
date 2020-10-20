#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("PermuteModel");
    mv::OpModel& om = unit.model();

    auto from_order = mv::Order("NCHW");
    auto to_order = mv::Order("NHWC");

    auto input0 = om.input("input0", {12,12,512,1}, mv::DType("Float16"), from_order);
    auto permute0 = om.permute("", input0, to_order);
    om.output("", permute0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
