#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("PSROIPoolingModel");
    mv::OpModel& om = unit.model();

    // Define Params
    size_t pooled_w = 7;
    size_t pooled_h = 7;
    size_t output_dim = 8;
    size_t group_size = 7;
    double spatial_scale = 0.0625;
    size_t spatial_bins_x = 1;
    size_t spatial_bins_y = 1;
    size_t num_rois = 5;
    std::string mode = "average";

    std::vector<uint16_t> weightsData(5*5);

    // Define tensors
    auto input0 = om.input({14,14,pooled_w*pooled_h*output_dim,1}, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{},{}}, "input0");

    //Load weights from file
    std::string  weights_filename(mv::utils::projectRootPath() + "/tests/layer/psroi_pooling/psroi_pooling.in2");
    std::ifstream w_file;
    w_file.open(weights_filename, std::fstream::in | std::fstream::binary);
    w_file.read((char*)(weightsData.data()), 5*num_rois * sizeof(uint16_t));
    std::vector<int64_t> weightsData_converted(5*num_rois);
    for(unsigned i = 0; i < weightsData.size(); ++i) {
        weightsData_converted[i] = weightsData[i];
    }

    auto weights0 = om.constantInt(weightsData_converted,{1,1,5,num_rois}, mv::DType("Float16"),
                                   mv::Order::getZMajorID(4), {{0},{1.},{},{}}, "weights0");

    // Build inputs vector
    std::vector<mv::Data::TensorIterator> inputs;
    inputs.push_back(input0);
    inputs.push_back(weights0);
    // Build Model
    auto psroiPooling = om.pSROIPooling(inputs, output_dim, group_size, spatial_scale, pooled_h, pooled_w,
                                        spatial_bins_x, spatial_bins_y, mode, mv::DType("Float16"));

    om.output(psroiPooling);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.compilationDescriptor().remove("adapt", "PostTrainingQuantize");

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
