#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("ResampleModel");
    mv::OpModel& om = unit.model();

    auto interpolation = std::string("NEAREST");
    bool antialias = false;
    int factor = 2;

    // Calculate output shape
    auto input_shape = mv::Shape({128,128,3,1});
    auto output_shape = input_shape;
    for (auto i=0; i<2; i++){
        output_shape[i] *= factor;
    }

    auto input0 = om.input("input0", input_shape, mv::DType("Float16"), mv::Order::getZMajorID(4));
    auto resample0 = om.resample("", input0, interpolation, antialias, output_shape);
    om.output("", resample0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
