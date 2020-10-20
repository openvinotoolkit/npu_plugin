#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("ArgmaxModel");
    mv::OpModel& om = unit.model();

    auto inputShape = mv::Shape({16,16,10,1});
    auto input0 = om.input("input:0#4", inputShape, mv::DType("Float16"), mv::Order::getZMajorID(4));
    input0->setQuantParams({{0},{1.0},{},{}});

    int out_max_val = 0;
    int top_k = 1;

    // Case #1 - axis=1
    int axis = 1;

    auto argmax0 = om.argmax("", input0, out_max_val, top_k, axis);
    om.output("", argmax0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}

