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
    num_classes = 21
    share_location = 1
    background_label_id = 0
    nms_threshold = 0.0
    top_k = 400
    code_type = "CENTER_SIZE"
    keep_top_k = 200
    confidence_threshold = 0.6
    variance_encoded_in_target = 0
    num_priors = 8732
    clip_before_nms = 0
    clip_after_nms = 0
    decrease_label_id = 0
    input_width = -1
    input_height = -1
    normalized = 1
    objectness_score = -1

    // Define tensors
    auto input0 = om.input({34928,1,1,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "input0");


    std::string compDescPath = path + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
