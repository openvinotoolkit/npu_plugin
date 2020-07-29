#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("Fp16Input");
    mv::OpModel& om = unit.model();
    static const auto inf = std::numeric_limits<double>::infinity();

    // auto input0 = om.input({4,4,16,1}, mv::DType("Float16"), mv::Order::getZMajorID(4),  {{0},{1.0},{},{}}, "input#01");


    auto data_0 = om.input({4,4,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4) /*NHWC*/,  {{0},{0.00392156862745098},{0},{1},{0},{1}}, "input");

    mv::Shape kernel = mv::Shape({1,1,16,16});
    std::vector<int64_t> weightsData0(kernel.totalSize(), 16);

    auto weights0 = om.constantInt(weightsData0,kernel, mv::DType("UInt8"), mv::Order("NCHW"), {{0},{0.00392156862745098},{-1.000000000000000},{1.000000000000000}}, "weights_conv");
    auto conv0 = om.conv(data_0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Default"),{{0},{1},{-inf},{inf},{0},{1}} , "conv");

    //Load weights from file
    std::string  weights_filename(mv::utils::projectRootPath() + "/tests/layer/deconv_fp16_small_tensor/weights.dat");
    std::ifstream w_file;
    std::vector<uint16_t> weightsData(2*2*16*16);
    w_file.open(weights_filename, std::fstream::in | std::fstream::binary);
    w_file.read((char*)(weightsData.data()), 2*2*16*16 * sizeof(uint16_t)); // WHIO
    std::vector<int64_t> weightsData_converted(2*2*16*16);
    for(unsigned i = 0; i < weightsData.size(); ++i) {
        weightsData_converted[i] = weightsData[i];
    }
    auto weights1 = om.constantInt(weightsData_converted, {2,2,16,16}, mv::DType("Float16"),
                                   mv::Order::getZMajorID(4), {{0},{1.},{},{}}, "dw_deconv_weights");

    auto deconv = om.deconv(conv0, weights1, {2, 2}, {0, 0, 0, 0}, 1, 1, false, mv::DType("Float16"), {{0},{1.0},{},{}}, "deconv_upscaling");

    const auto input_min = om.constant({0.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "input_min");
    const auto input_max = om.constant({16.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "input_max");
    const auto output_min = om.constant({0.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "output_min");
    const auto output_max = om.constant({16.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "output_max");
    auto fakeQuant = om.fakeQuantize(deconv, input_min, input_max, output_min, output_max, 256, "fakeQuantize");
    
    om.output(fakeQuant);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    // unit.compilationDescriptor().remove("adapt", "PostTrainingQuantize");
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}