#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("PriorBoxModel");
    mv::OpModel& om = unit.model();

    // Define Params
    auto flip = 1;
    auto clip = 0;
    auto step_w = 0.0;
    auto step_h = 0.0;
    auto offset = 0.5;

    std::vector<double> scalesData(10*10, 0);
    std::vector<double> min_sizesData({105.0});
    std::vector<double> max_sizesData({150.0});
    std::vector<double> aspect_ratiosData({2.0,3.0});
    std::vector<double> variancesData({0.1,0.1,0.2,0.2});

    //define tensors
    auto input0 = om.input({300,300,1,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "input:0#1");
    auto scales = om.constant(scalesData, {10,10,1,1}, mv::DType("Float64"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "scales");
    auto min_sizes = om.constant(min_sizesData, {1, min_sizesData.size(), 1, 1}, mv::DType("Float64"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "min_sizes");
    auto max_sizes = om.constant(max_sizesData, {1, max_sizesData.size(), 1, 1}, mv::DType("Float64"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "max_sizes");
    auto aspect_ratios = om.constant(aspect_ratiosData, {1, aspect_ratiosData.size(), 1, 1}, mv::DType("Float64"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "aspect_ratios");
    auto variances = om.constant(variancesData, {1, variancesData.size(), 1, 1}, mv::DType("Float64"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "variances");

    //build inputs vector
    std::vector<mv::Data::TensorIterator> inputs;
    inputs.push_back(input0);
    inputs.push_back(scales);
    inputs.push_back(min_sizes);
    inputs.push_back(max_sizes);
    inputs.push_back(aspect_ratios);
    inputs.push_back(variances);

    auto priorbox0 = om.priorbox(inputs, flip, clip, step_w, step_h, offset, mv::DType("Float16"));
    om.output(priorbox0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_MC-Prefetch1.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

}
