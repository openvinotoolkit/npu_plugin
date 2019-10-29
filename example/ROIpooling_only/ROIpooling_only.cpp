//This file is the parsed network which is created through python.
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include "iostream"
#include "fstream"

int main()
{
    std::string path = std::getenv("MCM_HOME");
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();

    // Define Params
    unsigned pooled_w = 0;
    unsigned pooled_h = 0;
    double spatial_scale = 0;
    unsigned roi_pooling_method = 0;
    unsigned num_rois = 0;

    // Define tensors
    //TODO: do something to get the other input, since multi-input
    auto input0 = om.input({125,13,13,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "input0");

    // Build inputs vector
    std::vector<mv::Data::TensorIterator> inputs;
    inputs.push_back(input0);
    // Build Model
    auto rOIPooling0 = om.rOIPooling(inputs, pooled_w, pooled_h, spatial_scale, roi_pooling_method, num_rois, mv::DType("Float16"));
    om.output(rOIPooling0);

    std::string compDescPath = path + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}