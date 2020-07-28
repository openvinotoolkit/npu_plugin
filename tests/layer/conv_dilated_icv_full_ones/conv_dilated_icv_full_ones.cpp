#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>
#include <include/mcm/op_model.hpp>

#include <limits>

int main()
{
    mv::CompilationUnit unit("ConvDilation");
    mv::OpModel& om = unit.model();

    static const auto inf = std::numeric_limits<double>::infinity();

    auto data_0 = om.input({16,16,16,1}, mv::DType("Float16"), mv::Order::getZMajorID(4),  {{0},{1.0},{},{}}, "input");

    //Kernel 3x3 - 16 input channel ; 16 output channels
    // 1 1 1
    // 1 1 1
    // 1 1 1
    // 1 1 1

    mv::Shape kernel = mv::Shape({2,2,16,16});
    std::vector<int64_t> weightsData0(kernel.totalSize(), 255);
    int64_t constant = 0x3C00; 

    for(int i = 0; i < 16; i++)
        for(int j = 0; j < 16; j++)
        {
            for(int k = 0; k < 4; k++)
            {
                // weightsData0[i*2*2*16 + j*2*2 + k] = 255 / (k + 1);
                weightsData0[i*2*2*16 + j*2*2 + k] = constant;
            }
        }

    auto weights0 = om.constantInt(weightsData0,kernel, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.},{},{}}, "weights_conv");

    // The dilation factor is 2
    // The padding is 2,2,2,2
    // This is SAME padding notation
    // Therefore subconvs padding will be 1,1,1,1 (as in slide 3 of design)

    // auto conv0 = om.conv(data_0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1,  mv::DType("UInt8"),{{0},{0.5647058823529412},{0},{144},{0},{1}} , "conv");
    auto conv0 = om.deconv(data_0, weights0, {2, 2}, {0, 0, 0, 0}, 1, 1, false, mv::DType("Float16"),{{0},{1.0},{},{}} , "conv");
    // std::vector<int64_t> biasData0 = mv::utils::generateSequence<int64_t> (16, 0, 0);
    // auto bias0 = om.constantInt(biasData0, {16}, mv::DType("Int32"), mv::Order::getRowMajorID(1), {{0}, {std::pow(0.00392156862745098,2)}, {}, {}});
    // auto bias = om.bias(conv0, bias0, mv::DType("UInt8"), {{0}, {0.000000768935}, {}, {}});

    om.output(conv0,mv::DType("Float16"), {{},{},{},{}});

    // std::vector<uint16_t> weightsData(2*2*16*32);
    // w_file.open(weights_filename, std::fstream::in | std::fstream::binary);
    // w_file.read((char*)(weightsData.data()), 2*2*16*32 * sizeof(uint16_t)); // WHIO
    // std::vector<int64_t> weightsData_converted(2*2*16*32);
    // for(unsigned i = 0; i < weightsData.size(); ++i) {
    //     weightsData_converted[i] = weightsData[i];
    // }
    // auto weights0 = om.constantInt(weightsData_converted, {2,2,16,32}, mv::DType("Float16"),
    //                                mv::Order::getZMajorID(4), {{0},{1.},{},{}}, "dw_deconv_weights");

    // auto deconv = om.deconv(input0, weights0, {2, 2}, {0, 0, 0, 0}, 1, 1, false, mv::DType("Float16"), {{0},{1.0},{},{}}, "deconv_upscaling");

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}

int main()
{
    mv::CompilationUnit unit("ConvDilation");
    mv::OpModel& om = unit.model();

    static const auto inf = std::numeric_limits<double>::infinity();

    auto data_0 = om.input({16,16,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4) /*NHWC*/,  {{0},{0.00392156862745098},{0},{1},{0},{1}}, "input");

    //Kernel 3x3 - 16 input channel ; 16 output channels
    // 1 1 1
    // 1 1 1
    // 1 1 1
    // 1 1 1

    mv::Shape kernel = mv::Shape({2,2,16,16});
    std::vector<int64_t> weightsData0(kernel.totalSize(), 255);

    // for(int i = 0; i < 16; i++)
    //     for(int j = 0; j < 16; j++)
    //     {
    //         for(int k = 0; k < 4; k++)
    //         {
    //             weightsData0[i*2*2*16 + j*2*2 + k] = 255 / (k + 1);
    //         }
    //     }

    auto weights0 = om.constantInt(weightsData0,kernel, mv::DType("UInt8"), mv::Order("NCHW"), {{0},{0.00392156862745098},{-1.000000000000000},{1.000000000000000}}, "weights_conv");

    // The dilation factor is 2
    // The padding is 2,2,2,2
    // This is SAME padding notation
    // Therefore subconvs padding will be 1,1,1,1 (as in slide 3 of design)

    // auto conv0 = om.conv(data_0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1,  mv::DType("UInt8"),{{0},{0.5647058823529412},{0},{144},{0},{1}} , "conv");
    auto conv0 = om.deconv(data_0, weights0, {2, 2}, {0, 0, 0, 0}, 1, 1, false, mv::DType("UInt8"),{{0},{0.5647058823529412},{0},{144},{0},{1}} , "conv");
    // std::vector<int64_t> biasData0 = mv::utils::generateSequence<int64_t> (16, 0, 0);
    // auto bias0 = om.constantInt(biasData0, {16}, mv::DType("Int32"), mv::Order::getRowMajorID(1), {{0}, {std::pow(0.00392156862745098,2)}, {}, {}});
    // auto bias = om.bias(conv0, bias0, mv::DType("UInt8"), {{0}, {0.000000768935}, {}, {}});

    om.output(conv0,mv::DType("UInt8"), {{},{},{},{}});

    // std::vector<uint16_t> weightsData(2*2*16*32);
    // w_file.open(weights_filename, std::fstream::in | std::fstream::binary);
    // w_file.read((char*)(weightsData.data()), 2*2*16*32 * sizeof(uint16_t)); // WHIO
    // std::vector<int64_t> weightsData_converted(2*2*16*32);
    // for(unsigned i = 0; i < weightsData.size(); ++i) {
    //     weightsData_converted[i] = weightsData[i];
    // }
    // auto weights0 = om.constantInt(weightsData_converted, {2,2,16,32}, mv::DType("Float16"),
    //                                mv::Order::getZMajorID(4), {{0},{1.},{},{}}, "dw_deconv_weights");

    // auto deconv = om.deconv(input0, weights0, {2, 2}, {0, 0, 0, 0}, 1, 1, false, mv::DType("Float16"), {{0},{1.0},{},{}}, "deconv_upscaling");

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
