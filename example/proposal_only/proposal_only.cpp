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
    double inf = std::numeric_limits<double>::infinity();

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
    std::vector<double> imageInfoData({224.0,224.0,1.0});
    std::vector<double> scaleData({0.25,0.5,1.0,2.0});
    std::vector<double> ratioData({0.5,1.0,2.0});
    // Define tensors
    std::string bboxPredPath = path + "/example/proposal_only/proposal.in2";
    std::vector<double> bboxPredData;
    std::fstream fs;
    double data;
    fs.open(bboxPredPath, std::fstream::in);
    while( fs >> data ) {
        bboxPredData.push_back(data);
    }
    fs.close();
    auto cls_pred = om.input({14,14,24,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "cls_pred0");
    auto bbox_pred = om.constant(bboxPredData,{14,14,48,1}, mv::DType("Float64"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "bbox_pred0");
    auto im_info = om.constant(imageInfoData, {1,1,1,imageInfoData.size()}, mv::DType("Float64"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "im_info0");
    auto scale = om.constant(scaleData, {1,1,1,scaleData.size()}, mv::DType("Float64"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "scale0");
    auto ratio = om.constant(ratioData, {1,1,1,ratioData.size()}, mv::DType("Float64"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "ratio0");
printf("3\n");
    // Build inputs vector
    std::vector<mv::Data::TensorIterator> inputs;
    inputs.push_back(cls_pred);
    inputs.push_back(bbox_pred);
    inputs.push_back(im_info);
    inputs.push_back(scale);
    inputs.push_back(ratio);
    // Build Model
    auto proposal0 = om.proposal(inputs, base_size, pre_nms_topn, post_nms_topn, nms_thresh, feat_stride, min_size, pre_nms_thresh, clip_before_nms, clip_after_nms, normalize, box_size_scale, box_coordinate_scale, framework, for_deformable, mv::DType("Float16"));
    om.output(proposal0);
printf("2\n");
    std::string compDescPath = path + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
printf("1\n");
    unit.run();
printf("0\n");
}
