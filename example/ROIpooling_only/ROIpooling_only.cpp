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
    std::vector<uint16_t> weightsData(5*5);

    // Define tensors
    auto input0 = om.input({14,14,32,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "input0");
    
    //Load weights from file
    std::string  weights_filename(path + "/example/ROIpooling_only/ROIpooling.in2");
    std::ifstream w_file;
    w_file.open(weights_filename, std::fstream::in | std::fstream::binary);
    w_file.read((char*)(weightsData.data()), 5*5 * sizeof(uint16_t));
    std::vector<int64_t> weightsData_converted(5*5);
    for(unsigned i = 0; i < weightsData.size(); ++i)
        weightsData_converted[i] = weightsData[i];

    auto weights0 = om.constantInt(weightsData_converted,{1,1,5*5,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.1524552064656746e-05},{-inf},{inf}}, "weights0");

    // Build inputs vector
    std::vector<mv::Data::TensorIterator> inputs;
    inputs.push_back(input0);
    inputs.push_back(weights0);
    // Build Model
    auto rOIPooling0 = om.rOIPooling(inputs, pooled_w, pooled_h, spatial_scale, roi_pooling_method, num_rois, mv::DType("Float16"));
    om.output(rOIPooling0);

    std::string compDescPath = path + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}