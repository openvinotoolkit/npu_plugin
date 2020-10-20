#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("SliceModel");
    mv::OpModel& om = unit.model();

    auto inputShape = mv::Shape({24,24,15,1});
    auto inputOrder = mv::Order("NHWC");
    auto axis = 1;
    auto num_splits = 5;
    // Calculate output begin & size Shapes 
    std::vector<mv::Shape> beginShapes;
    std::vector<mv::Shape> outputShapes;
    for (auto i=0; i<num_splits; ++i)
    {
        auto split_dim = (inputShape.ndims()-1 - axis);
        auto size = inputShape[split_dim] / num_splits;
        beginShapes.push_back(mv::Shape({0,0,size*i,0}));
        outputShapes.push_back(mv::Shape({inputShape[0],inputShape[1],size,1}));
    }

    // Input
    auto input0 = om.input("input0", inputShape, mv::DType("UInt8"), inputOrder);
    input0->setQuantParams({{0},{1.0},{},{}});

    // Slices
    std::vector<mv::Data::TensorIterator> slices;
    std::vector<mv::Data::TensorIterator> maxpools;
    for (auto i=0; i<num_splits; ++i)
    {
        auto slice = om.slice("slice" + std::to_string(i), input0, beginShapes.at(i), outputShapes.at(i));
        auto maxpool = om.maxPool("identity_maxpool" + std::to_string(i), slice, {1,1}, {1,1}, {0,0,0,0}, true);
        slice->setQuantParams({{0},{1.0},{},{}});
        maxpool->setQuantParams({{0},{1.0},{},{}});

        slices.push_back(slice);
        maxpools.push_back(maxpool);
    }

    // Concat
    std::string concat_axis = "C";
    auto concat0 = om.concat("concat0", maxpools, concat_axis);

    // Output
    auto output0 = om.output("", concat0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb-sc.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}

