#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>
int main()
{
    mv::CompilationUnit unit("ProposalModel");
    mv::OpModel& om = unit.model();

    // Define Params
    int base_size = 16;
    int pre_nms_topn = 6000;
    int post_nms_topn = 300;
    double nms_thresh = 0.7;
    int feat_stride = 16;
    int min_size = 16;
    float pre_nms_thresh = 0.5;
    bool clip_before_nms = true;
    bool clip_after_nms = false;
    bool normalize = false;
    float box_size_scale = 1.0;
    float box_coordinate_scale = 1.0;
    std::string framework = "CAFFE";
    bool for_deformable = false;
    std::vector<double> scaleData({8.0,16.0,32.0});
    std::vector<double> ratioData({0.5,1.0,2.0});
    std::vector<uint16_t> weightsData(14*14*36);
    std::vector<uint16_t> imInfoData(4);

    // Load weights from file
    std::string Weights_filename(mv::utils::projectRootPath() + "/tests/layer/proposal_caffe/proposal_caffe.in2");
    std::ifstream w_file;
    w_file.open(Weights_filename, std::fstream::in | std::fstream::binary);
    w_file.read((char*)(weightsData.data()), 14*14*36 * sizeof(uint16_t));
    std::vector<int64_t> weightsData_converted(14*14*36);
    for(unsigned i = 0; i < weightsData.size(); ++i)
        weightsData_converted[i] = weightsData[i];

    // Load imInfo from file
    std::string imInfo_filename(mv::utils::projectRootPath() + "/tests/layer/proposal_caffe/proposal_caffe.in3");
    std::ifstream i_file;
    i_file.open(imInfo_filename, std::fstream::in | std::fstream::binary);
    i_file.read((char*)(imInfoData.data()), 4 * sizeof(uint16_t));
    std::vector<int64_t> imInfoData_converted(4);
    for(unsigned i = 0; i < imInfoData.size(); ++i)
        imInfoData_converted[i] = imInfoData[i];

    auto cls_pred = om.input({14,14,18,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{},{}}, "cls_pred0");
    auto weights = om.constantInt(weightsData_converted, {14, 14, 36, 1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{},{}}, "weights");
    auto im_info = om.constantInt(imInfoData_converted, {1,4,1,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{},{}}, "im_info0");
    auto scale = om.constant(scaleData, {1,scaleData.size(),1,1}, mv::DType("Float64"), mv::Order::getZMajorID(4), {{0},{1.0},{},{}}, "scale0");
    auto ratio = om.constant(ratioData, {1,ratioData.size(),1,1}, mv::DType("Float64"), mv::Order::getZMajorID(4), {{0},{1.0},{},{}}, "ratio0");
    // Build inputs vector
    std::vector<mv::Data::TensorIterator> inputs;
    inputs.push_back(cls_pred);
    inputs.push_back(weights);
    inputs.push_back(im_info);
    inputs.push_back(scale);
    inputs.push_back(ratio);
    // Build Model
    auto proposal0 = om.proposal(inputs, base_size, pre_nms_topn, post_nms_topn, nms_thresh, feat_stride, min_size, pre_nms_thresh, clip_before_nms, clip_after_nms, normalize, box_size_scale, box_coordinate_scale, framework, for_deformable, mv::DType("Float16"));
    om.output(proposal0);
    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_MC-Prefetch1.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.compilationDescriptor().remove("adapt", "PostTrainingQuantize");
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}