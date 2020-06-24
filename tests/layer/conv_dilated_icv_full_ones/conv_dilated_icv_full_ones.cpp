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

    auto data_0 = om.input({16,16,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4) /*NHWC*/,  {{0},{0.00392156862745098},{0},{1},{0},{1}}, "input");

    //Kernel 3x3 - 16 input channel ; 16 output channels
    // 1 1 1
    // 1 1 1
    // 1 1 1
    // 1 1 1

    mv::Shape kernel = mv::Shape({3,3,16,16});
    std::vector<int64_t> weightsData0(kernel.totalSize(), 255);

    auto weights0 = om.constantInt(weightsData0,kernel, mv::DType("UInt8"), mv::Order("NCHW"), {{0},{0.00392156862745098},{-1.000000000000000},{1.000000000000000}}, "weights_conv");

    // The dilation factor is 2
    // The padding is 2,2,2,2
    // This is SAME padding notation
    // Therefore subconvs padding will be 1,1,1,1 (as in slide 3 of design)

    auto conv0 = om.conv(data_0, weights0, {1, 1}, {2, 2, 2, 2}, 2, 1,  mv::DType("UInt8"),{{0},{0.5647058823529412},{0},{144},{0},{1}} , "conv");
//    auto conv0 = om.conv(data_0, weights0, {1, 1}, {1,1,1,1}, 1, 1,  mv::DType("UInt8"),{{0},{0.5647058823529412},{0},{144},{0},{1}} , "conv");

    // Identidy conv - this should not change output of dilated conv
    // Output of dilated Conv is all 3f's

    mv::Shape kernel1 = mv::Shape({1,1,16,16});
    std::vector<int64_t> weightsData1(kernel1.totalSize(), 0);

    for (unsigned k = 0; k < kernel1[mv::KERNEL_OUTPUT_CHANNELS]; ++k)
    {
        for (unsigned c = 0; c < kernel1[mv::KERNEL_INPUT_CHANNELS]; ++c)
        {
            for (unsigned h = 0; h < kernel1[mv::KERNEL_HEIGHT]; ++h)
            {
                for (unsigned w = 0; w < kernel1[mv::KERNEL_WIDTH]; ++w)
                {
                    const size_t idx = (k * kernel1[mv::KERNEL_INPUT_CHANNELS] * kernel1[mv::KERNEL_WIDTH] * kernel1[mv::KERNEL_HEIGHT]) +
                                       (c * kernel1[mv::KERNEL_WIDTH] * kernel1[mv::KERNEL_HEIGHT]) +
                                       (h * kernel1[mv::KERNEL_WIDTH]) +
                                        w;
                    if (c == k)
                        weightsData1[idx] = 255;
                    else
                        weightsData1[idx] = 0;
                }
            }
        }
    }

    auto weights1 = om.constantInt(weightsData1,kernel1, mv::DType("UInt8"), mv::Order("NCHW"), {{0},{0.00392156862745098},{-1.000000000000000},{1.000000000000000}}, "weights_conv1");
    auto conv1 = om.conv(conv0, weights1, {1, 1}, {0, 0, 0, 0}, 1, 1,  mv::DType("UInt8"),{{0},{0.5647058823529412},{-inf},{inf},{0},{1}} , "conv1");

    om.output(conv1,mv::DType("UInt8"), {{},{},{},{}});

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
