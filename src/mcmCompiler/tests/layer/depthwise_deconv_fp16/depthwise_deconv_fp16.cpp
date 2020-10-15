#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("Fp16Input");
    mv::OpModel& om = unit.model();

    auto input0 = om.input({4,4,16,1}, mv::DType("Float16"), mv::Order::getZMajorID(4),  {{0},{1.0},{},{}}, "input#01");

    //Load weights from file
    std::string  weights_filename(mv::utils::projectRootPath() + "/tests/layer/depthwise_deconv_fp16/depthwise_deconv_fp16.in2");
    std::ifstream w_file;
    std::vector<uint16_t> weightsData(2*2*16*1);
    w_file.open(weights_filename, std::fstream::in | std::fstream::binary);
    w_file.read((char*)(weightsData.data()), 2*2*16*1 * sizeof(uint16_t)); // WHIO
    std::vector<int64_t> weightsData_converted(2*2*16*1);
    for(unsigned i = 0; i < weightsData.size(); ++i) {
        weightsData_converted[i] = weightsData[i];
    }
    auto weights0 = om.constantInt(weightsData_converted, {2,2,16,1}, mv::DType("Float16"),
                                   mv::Order::getZMajorID(4), {{0},{1.},{},{}}, "dw_deconv_weights");

    auto deconv = om.deconv(input0, weights0, {2, 2}, {0, 0, 0, 0}, 1, 16, true, mv::DType("Float16"), {{0},{1.0},{},{}}, "deconv_upscaling");
    om.output(deconv);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.compilationDescriptor().remove("adapt", "PostTrainingQuantize");
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}