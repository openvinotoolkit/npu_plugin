#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

// Note: 2 pairs of inputs/output data are available. 
// 1st input shape: 7x167x271 (CHW) with region = across, local_size = 5
// 2nd input shape: 96x86x86 (CHW) with region = same, local_size = 3
// example is hardcoded for 2nd case

int main()
{

    mv::CompilationUnit unit("normOnlyModel");
    mv::OpModel& om = unit.model();
    //auto input0 = om.input({271,167,7,1}, mv::DType("Float64"), mv::Order::getZMajorID(4), {{0},{1.0},{},{}}, "input:0#4");
    auto input0 = om.input({86,86,96,1}, mv::DType("Float64"), mv::Order::getZMajorID(4), {{0},{1.0},{},{}}, "input:0");
    float alpha = 1;
    float beta = 0.75;
    //unsigned local_size = 5;
    unsigned local_size = 3;
    //std::string region = "across";
    std::string region = "same";
    auto norm0 = om.norm(input0, alpha, beta, region, local_size);
    om.output(norm0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_MC-Prefetch1.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

}
