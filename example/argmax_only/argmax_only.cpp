#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include "iostream"
#include "fstream"

int main()
{
    std::string path = std::getenv("MCM_HOME");

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();

    auto inputShape = mv::Shape({16,16,10,1});
    auto input0 = om.input(inputShape, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "input:0#4");

    int out_max_val = 0;
    int top_k = 1;

    // Case #1 - axis=1
    int axis = 1;

    // Case #2 - no axis
    //int axis = 99;

    // Case #3 - invalid axis
    //int axis = -4;

    auto argmax0 = om.argmax(input0, out_max_val, top_k, axis, mv::DType("Float16"));
    om.output(argmax0);

    std::string compDescPath = path + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}

