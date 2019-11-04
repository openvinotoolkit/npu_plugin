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

    // Define Params
    unsigned pooled_w = 7;
    unsigned pooled_h = 7;
    double spatial_scale = 0.0625;
    unsigned roi_pooling_method = 1;
    unsigned num_rois = 5;

    // Define tensors
    auto input0 = om.input({14,14,32,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "input0");
    std::string weightsPath = path + "/example/ROIpooling_only/ROIpooling.in2";
    std::vector<double> scaleWeights0;
    double weight;
    std::fstream fs;
    fs.open(weightsPath, std::fstream::in);
    while( fs >> weight ) {
        scaleWeights0.push_back(weight);
    }
    fs.close();
    auto scales0 = om.constant(scaleWeights0,{5,5,1,1}, mv::DType("Float64"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "scale_weights#0");
    // Build inputs vector
    std::vector<mv::Data::TensorIterator> inputs;
    inputs.push_back(input0);
    inputs.push_back(scales0);
    // Build Model
    auto rOIPooling0 = om.rOIPooling(inputs, pooled_w, pooled_h, spatial_scale, roi_pooling_method, num_rois, mv::DType("Float16"));
    om.output(rOIPooling0);

    std::string compDescPath = path + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}