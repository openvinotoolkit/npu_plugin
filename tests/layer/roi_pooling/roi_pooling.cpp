#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("ROIPoolingModel");
    mv::OpModel& om = unit.model();

    // Define Params
    unsigned pooled_w = 7;
    unsigned pooled_h = 7;
    double spatial_scale = 0.0625;
    unsigned roi_pooling_method = 1;
    unsigned num_rois = 5;
    std::vector<uint16_t> weightsData(5*5);

    // Define tensors
    auto input0 = om.input({14,14,32,1}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{},{}}, "input0");

    //Load weights from file
    std::string  weights_filename(mv::utils::projectRootPath() + "/tests/layer/roi_pooling/roi_pooling.in2");
    std::ifstream w_file;
    w_file.open(weights_filename, std::fstream::in | std::fstream::binary);
    w_file.read((char*)(weightsData.data()), 5*5 * sizeof(uint16_t));
    std::vector<int64_t> weightsData_converted(5*5);
    for(unsigned i = 0; i < weightsData.size(); ++i)
        weightsData_converted[i] = weightsData[i];

    auto weights0 = om.constantInt(weightsData_converted,{1,1,5*5,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.1524552064656746e-05},{},{}}, "weights0");

    // Build inputs vector
    std::vector<mv::Data::TensorIterator> inputs;
    inputs.push_back(input0);
    inputs.push_back(weights0);
    // Build Model
    auto rOIPooling0 = om.rOIPooling(inputs, pooled_w, pooled_h, spatial_scale, roi_pooling_method, num_rois, mv::DType("Float16"));
    om.output(rOIPooling0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_MC-Prefetch1.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.compilationDescriptor().remove("adapt", "PostTrainingQuantize");

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}