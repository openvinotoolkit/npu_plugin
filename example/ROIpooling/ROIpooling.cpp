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
    auto input0 = om.input({125,13,13,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "input0");


    auto rOIPooling = om.rOIPooling

    auto reorgyolo0 = om.reorgYolo(input0, stride, mv::DType("Float16"));
    om.output(reorgyolo0);

    //input shape: {125, 13, 13, 1}
    //output shape: {21125, 1}

    std::string compDescPath = path + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}

rOIPooling(const std::vector< Data::TensorIterator >& inputs, 
    const unsigned& pooled_w, const unsigned& pooled_h, 
    const double& spatial_scale, 
    const unsigned& roi_pooling_method, 
    const unsigned& num_rois, const DType& dType = mv::DType("Default"), 
    const mv::QuantizationParams& quantParams = {{},{},{},{}}, 
    const std::string& name = "") = 0;