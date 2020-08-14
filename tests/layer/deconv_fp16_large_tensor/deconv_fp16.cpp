#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("Fp16Input");
    mv::OpModel& om = unit.model();
    static const auto inf = std::numeric_limits<double>::infinity();

    // auto input0 = om.input({4,4,16,1}, mv::DType("Float16"), mv::Order::getZMajorID(4),  {{0},{1.0},{},{}}, "input#01");


    // auto data_0 = om.input({30,23,1024,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4) /*NHWC*/,  {{0},{0.00392156862745098},{0},{1},{0},{1}}, "input");

    mv::Shape kernel = mv::Shape({1,1,1024,1024});
    std::vector<int64_t> weightsData0(kernel.totalSize(), 1);

    auto weights0 = om.constantInt(weightsData0,kernel, mv::DType("UInt8"), mv::Order("NCHW"), {{0},{0.00392156862745098},{-1.000000000000000},{1.000000000000000}}, "weights_conv");
    // auto conv0 = om.conv(data_0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Default"),{{0},{1},{-inf},{inf},{0},{1}} , "conv");

    // use fakequantize

    auto data_0 = om.input({30,23,1024,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4) /*NHWC*/,  {{0},{0.00392156862745098},{0},{1},{0},{1}}, "input");

    // const auto data_0_input_min = om.constant({0.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "data_0_input_min");
    // const auto data_0_input_max = om.constant({1.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "data_0_input_max");
    // const auto data_0_output_min = om.constant({0.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "data_0_output_min");
    // const auto data_0_output_max = om.constant({1.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "data_0_output_max");
    // auto data_0_fakeQuant = om.fakeQuantize(data_0, data_0_input_min, data_0_input_max, data_0_output_min, data_0_output_max, 256, "data_0_fakeQuant");

    // mv::Shape kernel = mv::Shape({1,1,1024,1024});
    // std::vector<double> weightsData0(kernel.totalSize(), 1.0 / 255);

    // auto weights0 = om.constant(weightsData0,kernel, mv::DType("Float32"), mv::Order("NCHW"), {{0},{1.0},{},{}}, "weights_conv");

    // const auto weights0_input_min = om.constant({0.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "weights0_input_min");
    // const auto weights0_input_max = om.constant({1.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "weights0_input_max");
    // const auto weights0_output_min = om.constant({0.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "weights0_output_min");
    // const auto weights0_output_max = om.constant({1.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "weights0_output_max");
    // auto weights0_fakeQuant = om.fakeQuantize(weights0, weights0_input_min, weights0_input_max, weights0_output_min, weights0_output_max, 256, "weights0_fakeQuant");

    auto conv0 = om.conv(data_0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Default"),{{0},{1},{-inf},{inf},{0},{1}} , "conv");

    vector<double> biasData(1024, 0.5);

    auto mvBiases = om.constant(biasData, {1024}, mv::DType("Float32"), mv::Order::getColMajorID(1));

    auto conv0_bias = om.bias(
        conv0, mvBiases, mv::DType("Default"), {{0},{1},{-inf},{inf},{0},{1}}, "conv_bias");

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

    std::vector<double> doubleWeights(2*2*1024*512, 2.0);
    auto weights1 = om.constant(doubleWeights, {2,2,1024,512}, mv::DType("Float32"),
                                   mv::Order("NCHW"), {{0},{1.},{},{}}, "dw_deconv_weights");

    auto deconv = om.deconv(conv0_bias, weights1, {2, 2}, {0, 0, 0, 0}, 1, 1, false, mv::DType("Float16"), {{0},{1.0},{},{}}, "deconv_upscaling");

    const auto input_min = om.constant({0.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "input_min");
    const auto input_max = om.constant({8192.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "input_max");
    const auto output_min = om.constant({0.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "output_min");
    const auto output_max = om.constant({8192.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "output_max");
    auto fakeQuant = om.fakeQuantize(deconv, input_min, input_max, output_min, output_max, 256, "fakeQuantize");
    
    om.output(fakeQuant, mv::DType("Float16"));

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    // unit.compilationDescriptor().remove("adapt", "PostTrainingQuantize");
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}