#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{
    // Input data for this test has to be generated using
    // ScaleShiftInput.py

    mv::CompilationUnit unit("UnfusableScaleModel");
    mv::OpModel& om = unit.model();

    auto input0 = om.input({4,4,3,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4),  {{0},{1.0},{},{}}, "input#170");
    std::vector<int64_t> scalesData0 = {1, 1, 1};

    auto scales0 = om.constantInt(scalesData0,{3}, mv::DType("UInt8"), mv::Order::getRowMajorID(1), {{0},{0.0174292, 0.017507,  0.0171248},{},{}});
    auto scale = om.scale(input0, scales0, mv::DType("UInt8"), {{118}, {0.0172290776471}, {}, {}}, "scale0");

    std::vector<int64_t> biasData0 = {-104, -116, -124};
    auto bias0 = om.constantInt(biasData0, {3}, mv::DType("Int32"), mv::Order::getRowMajorID(1), {{0}, {0.0174292, 0.017507,  0.0171248}, {}, {}});
    auto bias = om.bias(scale, bias0, mv::DType("UInt8"), {{118}, {0.0172290776471}, {}, {}});
    om.output(bias);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.compilationDescriptor().remove("adapt", "PostTrainingQuantize");
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

}
