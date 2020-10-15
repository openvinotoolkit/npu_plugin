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

    auto data_0 = om.input({32,32,512,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4) /*NHWC*/,  {{0},{0.00392156862745098},{-inf},{inf},{0},{1}}, "input");


    mv::Shape kernel = mv::Shape({3,3,512,1024});
    std::vector<int64_t> weightsData0(kernel.totalSize(), 255);

    auto weights0 = om.constantInt(weightsData0,kernel, mv::DType("UInt8"), mv::Order("NCHW"), {{0},{0.00001/255},{-inf},{inf}}, "weights_conv");
    auto conv0 = om.conv(data_0, weights0, {1, 1}, {6,6,6,6}, 6, 1,  mv::DType("UInt8"),{{0},{47.18592/255},{-inf},{inf},{0},{1}} , "conv");

//    mv::Shape weightsShape = mv::Shape({1,1,1024,1024});
//    std::vector<int64_t> weightsData1(weightsShape.totalSize(), 0);

//    for (unsigned k = 0; k < weightsShape[mv::KERNEL_OUTPUT_CHANNELS]; ++k)
//    {
//        for (unsigned c = 0; c < weightsShape[mv::KERNEL_INPUT_CHANNELS]; ++c)
//        {
//            for (unsigned h = 0; h < weightsShape[mv::KERNEL_HEIGHT]; ++h)
//            {
//                for (unsigned w = 0; w < weightsShape[mv::KERNEL_WIDTH]; ++w)
//                {
//                    const size_t idx = (k * weightsShape[mv::KERNEL_INPUT_CHANNELS] * weightsShape[mv::KERNEL_WIDTH] * weightsShape[mv::KERNEL_HEIGHT]) +
//                                       (c * weightsShape[mv::KERNEL_WIDTH] * weightsShape[mv::KERNEL_HEIGHT]) +
//                                       (h * weightsShape[mv::KERNEL_WIDTH]) +
//                                        w;
//                    if (c == k)
//                        weightsData1[idx] = 255;
//                    else
//                        weightsData1[idx] = 0;
//                }
//            }
//        }
//    }
//    auto weights1 = om.constantInt(weightsData1,weightsShape, mv::DType("UInt8"), mv::Order("NHWC"), {{0},{0.00392156862745098},{-inf},{inf}}, "weights_conv1");
//    auto conv1 = om.conv(conv0, weights1, {1, 1}, {0,0,0,0}, 1, 1,  mv::DType("UInt8"),{{0},{18504.282352941176},{-inf},{inf},{0},{1}} , "conv1");

    om.output(conv0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
