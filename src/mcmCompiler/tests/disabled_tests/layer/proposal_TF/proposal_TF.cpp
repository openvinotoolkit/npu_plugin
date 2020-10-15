#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>
int main()
{
    mv::CompilationUnit unit("ProposalModel");
    mv::OpModel& om = unit.model();

    // Define Params
    int base_size = 256;
    int pre_nms_topn = 2147483647;
    int post_nms_topn = 100;
    double nms_thresh = 0.7;
    int feat_stride = 16;
    int min_size = 10;
    float pre_nms_thresh = 0.0;
    bool clip_before_nms = true;
    bool clip_after_nms = false;
    bool normalize = false;
    float box_size_scale = 5.0;
    float box_coordinate_scale = 10.0;
    std::string framework = "TENSORFLOW";
    bool for_deformable = false;
    std::vector<float> scale({0.25,0.5,1.0,2.0});
    std::vector<float> ratio({0.5,1.0,2.0});
    std::vector<double> weightsData(14*14*48);
    std::vector<double> imInfoData(3);

    // Load weights from file
    std::string Weights_filename(mv::utils::projectRootPath() + "/tests/layer/proposal_TF/proposal_TF.in2");
    std::ifstream w_file;
    w_file.open(Weights_filename, std::fstream::in | std::fstream::binary);
    w_file.read((char*)(weightsData.data()), 14*14*48 * sizeof(uint16_t));
    std::vector<int64_t> weightsData_converted(14*14*48);
    for(unsigned i = 0; i < weightsData.size(); ++i)
        weightsData_converted[i] = weightsData[i];

    // Load imInfo from file
    std::string imInfo_filename(mv::utils::projectRootPath() + "/tests/layer/proposal_TF/proposal_TF.in3");
    std::ifstream i_file;
    i_file.open(imInfo_filename, std::fstream::in | std::fstream::binary);
    i_file.read((char*)(imInfoData.data()), 3 * sizeof(uint16_t));
    std::vector<int64_t> imInfoData_converted(3);
    for(unsigned i = 0; i < imInfoData.size(); ++i)
        imInfoData_converted[i] = imInfoData[i];

    auto cls_pred = om.input({14,14,24,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{},{}}, "cls_pred0");
    auto weights = om.constantInt(weightsData_converted, {14, 14, 48, 1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{},{}}, "weights");
    auto im_info = om.constantInt(imInfoData_converted, {1,3,1,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{},{}}, "im_info0");
    // Build inputs vector
    std::vector<mv::Data::TensorIterator> inputs;
    inputs.push_back(cls_pred);
    inputs.push_back(weights);
    inputs.push_back(im_info);
    // Build Model
    auto proposal = om.proposal(inputs, scale, ratio, base_size, pre_nms_topn, post_nms_topn, nms_thresh, feat_stride, min_size, pre_nms_thresh, clip_before_nms, clip_after_nms, normalize, box_size_scale, box_coordinate_scale, framework, for_deformable, mv::DType("Float16"));
    om.output(proposal);
    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.compilationDescriptor().remove("adapt", "PostTrainingQuantize");
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
