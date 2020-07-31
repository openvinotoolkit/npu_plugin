#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("Fp16Input");
    mv::OpModel& om = unit.model();
    static const auto inf = std::numeric_limits<double>::infinity();

    auto input0 = om.input({30,23,1024,1}, mv::DType("Float16"), mv::Order::getZMajorID(4),  {{0},{1.0},{},{}}, "input#01");

    //Load weights from file
    std::string  weights_filename(mv::utils::projectRootPath() + "/tests/layer/deconv_fp16_large_tensor/weights.dat");
    std::ifstream w_file;
    std::vector<uint16_t> weightsData(2*2*1024*512);
    w_file.open(weights_filename, std::fstream::in | std::fstream::binary);
    w_file.read((char*)(weightsData.data()), 2*2*1024*512 * sizeof(uint16_t)); // WHIO
    std::vector<int64_t> weightsData_converted(2*2*1024*512);
    for(unsigned i = 0; i < weightsData.size(); ++i) {
        weightsData_converted[i] = weightsData[i];
    }
    auto weights1 = om.constantInt(weightsData_converted, {2,2,1024,512}, mv::DType("Float16"),
                                   mv::Order::getZMajorID(4), {{0},{1.},{},{}}, "dw_deconv_weights");

    auto deconv = om.deconv(input0, weights1, {2, 2}, {0, 0, 0, 0}, 1, 1, false, mv::DType("Float16"), {{0},{1.0},{},{}}, "deconv_upscaling");

    const auto input_min = om.constant({0.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "input_min");
    const auto input_max = om.constant({1024.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "input_max");
    const auto output_min = om.constant({0.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "output_min");
    const auto output_max = om.constant({1024.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "output_max");
    auto fakeQuant = om.fakeQuantize(deconv, input_min, input_max, output_min, output_max, 256, "fakeQuantize");
    
    om.output(deconv);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    // unit.compilationDescriptor().remove("adapt", "PostTrainingQuantize");
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}