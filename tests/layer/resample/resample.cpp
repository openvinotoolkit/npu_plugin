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

    auto input0 = om.input(input_shape, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{},{}}, "input0");
    auto resample0 = om.resample(input0, interpolation, antialias, output_shape, mv::DType("Default"));
    om.output(resample0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_MC-Prefetch1.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

}
