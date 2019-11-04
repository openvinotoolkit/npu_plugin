//This file is the parsed network which is created through python.
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "build/meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include "iostream"
#include "fstream"

int main()
{
    std::string path = std::getenv("MCM_HOME");

    mv::CompilationUnit unit("parserModel");
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
    std::vector<double> scaleData({0.25,0.5,1.0,2.0});
    std::vector<double> ratioData({0.5,1.0,2.0});
    std::vector<uint16_t> weightsData(14*14*48);
    std::vector<uint16_t> imInfoData(3);

    // Load weights from file
    std::string Weights_filename(path + "/example/proposal_only/proposal.in2");
    std::ifstream w_file;
    w_file.open(Weights_filename, std::fstream::in | std::fstream::binary);
    w_file.read((char*)(weightsData.data()), 14*14*48 * sizeof(uint16_t));
    std::vector<int64_t> weightsData_converted(14*14*48);
    for(unsigned i = 0; i < weightsData.size(); ++i)
        weightsData_converted[i] = weightsData[i];

    // Load imInfo from file
    std::string imInfo_filename(path + "/example/proposal_only/proposal.in3");
    std::ifstream i_file;
    i_file.open(imInfo_filename, std::fstream::in | std::fstream::binary);
    i_file.read((char*)(imInfoData.data()), 3 * sizeof(uint16_t));
    std::vector<int64_t> imInfoData_converted(3);
    for(unsigned i = 0; i < imInfoData.size(); ++i)
        imInfoData_converted[i] = imInfoData[i];

    auto cls_pred = om.input({14,14,24,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "cls_pred0");
    auto weights = om.constantInt(weightsData_converted, {14, 14, 48, 1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "weights");
    auto im_info = om.constantInt(imInfoData_converted, {1,3,1,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "im_info0");
    auto scale = om.constant(scaleData, {1,scaleData.size(),1,1}, mv::DType("Float64"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "scale0");
    auto ratio = om.constant(ratioData, {1,ratioData.size(),1,1}, mv::DType("Float64"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "ratio0");
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
    std::string compDescPath = path + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
/*
int main()
{
    std::string path = std::getenv("MCM_HOME");

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();

    // Define Params
    int base_size = 16;
    int pre_nms_topn = 300;
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
    std::vector<double> scaleData({0.25,0.5,1.0,2.0});
    std::vector<double> ratioData({0.5,1.0,2.0});
    std::vector<uint16_t> weightsData(14*14*36);
    std::vector<uint16_t> imInfoData(4);

    // Load weights from file
    std::string Weights_filename(path + "/example/proposal_only/proposal.in2");
    std::ifstream w_file;
    w_file.open(Weights_filename, std::fstream::in | std::fstream::binary);
    w_file.read((char*)(weightsData.data()), 14*14*36 * sizeof(uint16_t));
    std::vector<int64_t> weightsData_converted(14*14*36);
    for(unsigned i = 0; i < weightsData.size(); ++i)
        weightsData_converted[i] = weightsData[i];

    // Load imInfo from file
    std::string imInfo_filename(path + "/example/proposal_only/proposal.in3");
    std::ifstream i_file;
    i_file.open(imInfo_filename, std::fstream::in | std::fstream::binary);
    i_file.read((char*)(imInfoData.data()), 4 * sizeof(uint16_t));
    std::vector<int64_t> imInfoData_converted(4);
    for(unsigned i = 0; i < imInfoData.size(); ++i)
        imInfoData_converted[i] = imInfoData[i];

    auto cls_pred = om.input({14,14,18,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "cls_pred0");
    auto weights = om.constantInt(weightsData_converted, {14, 14, 36, 1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "weights");
    auto im_info = om.constantInt(imInfoData_converted, {1,4,1,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "im_info0");
    auto scale = om.constant(scaleData, {1,scaleData.size(),1,1}, mv::DType("Float64"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "scale0");
    auto ratio = om.constant(ratioData, {1,ratioData.size(),1,1}, mv::DType("Float64"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "ratio0");
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
    std::string compDescPath = path + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}*/